// Copyright 2020 The TensorStore Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Extracts a slice of a volumetric dataset, outputtting it as a 2d jpeg image.
//
// extract_slice --output_file=/tmp/foo.jpg --input_spec=...
#include <unistd.h>
#include <stdint.h>
#include<typeinfo>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>
#include <chrono>
#include <future>
#include <cmath>
#include "tensorstore/tensorstore.h"
#include "tensorstore/context.h"
#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/driver/zarr/dtype.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/compression/blosc.h"
//#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/kvstore/memory/memory_key_value_store.h"
#include "tensorstore/open.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
#include "BS_thread_pool.hpp"
using ::tensorstore::Context;
using ::tensorstore::StrCat;
using tensorstore::Index;
using ::tensorstore::internal_zarr::ChooseBaseDType;
using namespace std::chrono_literals;

template <typename T>
std::unique_ptr<std::vector<T>> downsample_average(std::vector<T>& source_array, int row, int col) {
  int new_row = static_cast<int>(ceil(row / 2.0));
  int new_col = static_cast<int>(ceil(col / 2.0));
  auto result_data = std::make_unique<std::vector<T>>(new_row * new_col);
  auto data_ptr = result_data->data();

  int even_row{0}, even_col{0};
  
  if (row%2==0){
    even_row = row;
  }
  else {
    even_row = row-1;
  }

  if (col%2==0){
    even_col = col;
  }
  else {
    even_col = col-1;
  }

  for (int i = 0; i < even_row; i=i+2) {
    int row_offset = (i / 2) * new_col;
    int prev_row_offset = i * col;
    int prev_row_offset_2 = (i+1) * col;
    for (int j = 0; j < even_col; j = j + 2) {
      int new_data_index = row_offset + (j / 2);
      data_ptr[new_data_index] =   (source_array[prev_row_offset + j] 
                                   + source_array[prev_row_offset + j + 1]
                                   + source_array[prev_row_offset_2 + j]
                                   + source_array[prev_row_offset_2 + j + 1])*0.25;
    }
  }
  // fix the last col if odd
  if (col % 2 == 1) {
    for (int i = 0; i < even_row; i=i+2) {
      data_ptr[((i / 2)+1) * new_col-1] = 0.5*(source_array[(i+1)*col-1] + source_array[(i+2)*col-1]);
    }
  }

  // fix the last row if odd
  if (row % 2 == 1) {
    int col_offset = (new_row-1)*new_col;
    int old_col_offset =(row-1)*col;
    for (int i = 0; i < even_col; i=i+2) {
      data_ptr[col_offset+(i/2)] = 0.5*(source_array[old_col_offset+i] + source_array[old_col_offset+i+1]);
    }
  }
  
  // fix the last element if both row and col are odd
  if (row%2==1 && col%2==1){
      data_ptr[new_row*new_col-1] = source_array[row*col-1];
  }

  return result_data;
}

template <typename T, typename V>
double DownsampleAndWrtieChunk(T&& source, T&& dest, int x1, int x2, int y1, int y2, int x1_old, int x2_old, int y1_old, int y2_old){
  std::vector<V> read_buffer((x2_old-x1_old)*(y2_old-y1_old));
  auto array = tensorstore::Array(read_buffer.data(), {x2_old-x1_old, y2_old-y1_old}, tensorstore::c_order);

  tensorstore::Read(source | 
          tensorstore::Dims(0).ClosedInterval(x1_old,x2_old-1) |
          tensorstore::Dims(1).ClosedInterval(y1_old,y2_old-1) ,
          tensorstore::UnownedToShared(array)).value();

  auto resutl = downsample_average(read_buffer, (x2_old-x1_old), (y2_old-y1_old));
  auto result_array = tensorstore::Array(resutl->data(), {x2-x1, y2-y1}, tensorstore::c_order);
  tensorstore::Write(tensorstore::UnownedToShared(result_array), dest |
      tensorstore::Dims(0).ClosedInterval(x1,x2-1) |
      tensorstore::Dims(1).ClosedInterval(y1,y2-1)).value();     
  // wait for write
  return 0.0;
}

template <typename T>
double CopyChunk(T&& source, T&& dest, int x1, int x2, int y1, int y2)
{
  auto time1 = std::chrono::steady_clock::now();
  auto array = tensorstore::AllocateArray({x2-x1, y2-y1},tensorstore::c_order,
                                 tensorstore::value_init, source.dtype());
  // initiate a read
  tensorstore::Read(source | 
            tensorstore::Dims(0).ClosedInterval(0,0) |
            tensorstore::Dims(1).ClosedInterval(0,0) |
            tensorstore::Dims(2).ClosedInterval(0,0) |
            tensorstore::Dims(3).ClosedInterval(x1,x2-1) |
            tensorstore::Dims(4).ClosedInterval(y1,y2-1) ,
            array).value();
                     
  // initiate write
  tensorstore::Write(array, dest |
      tensorstore::Dims(0).ClosedInterval(x1,x2-1) |
      tensorstore::Dims(1).ClosedInterval(y1,y2-1)).value();     
  // wait for write

  auto time2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds =  time2-time1;

  return elapsed_seconds.count();
}

template <typename T>
double CopyChunk2(T&& source, T&& dest, int x1, int x2, int y1, int y2)
{
  auto time1 = std::chrono::steady_clock::now();

   tensorstore::Copy(source | 
            tensorstore::Dims(0).ClosedInterval(0,0) |
            tensorstore::Dims(1).ClosedInterval(0,0) |
            tensorstore::Dims(2).ClosedInterval(0,0) |
            tensorstore::Dims(3).ClosedInterval(x1,x2-1) |
            tensorstore::Dims(4).ClosedInterval(y1,y2-1) ,
       dest | 
      tensorstore::Dims(0).ClosedInterval(x1,x2-1) |
      tensorstore::Dims(1).ClosedInterval(y1,y2-1)).value();     
  
  auto time2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds =  time2-time1;

  return elapsed_seconds.count();
}

template <typename T>
double CopyChunk3(T source, T dest, int x1, int x2, int y1, int y2)
{
  auto time1 = std::chrono::steady_clock::now();
  auto array = tensorstore::AllocateArray({x2-x1, y2-y1},tensorstore::c_order,
                                 tensorstore::value_init, source.dtype());
  // initiate a read
  tensorstore::Read(source | 
            tensorstore::Dims(0).ClosedInterval(0,0) |
            tensorstore::Dims(1).ClosedInterval(0,0) |
            tensorstore::Dims(2).ClosedInterval(0,0) |
            tensorstore::Dims(3).ClosedInterval(x1,x2-1) |
            tensorstore::Dims(4).ClosedInterval(y1,y2-1) ,
            array).value();
                     
  // initiate write
  tensorstore::Write(array, dest |
      tensorstore::Dims(0).ClosedInterval(x1,x2-1) |
      tensorstore::Dims(1).ClosedInterval(y1,y2-1)).value();     
  // wait for write

  auto time2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds =  time2-time1;

  return elapsed_seconds.count();
}

void write_base_scale(std::string input_file, std::string output_file){

  int chunk_size = 1024;
  BS::thread_pool pool(4);
  tensorstore::Context context = Context::Default();
  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store1, tensorstore::Open({{"driver", "ometiff"},

                            {"kvstore", {{"driver", "tiled_tiff"},
                                         {"path", input_file}}
                            }
                            },
                            //context,
                            tensorstore::OpenMode::open,
                            tensorstore::ReadWriteMode::read).result());

  auto shape = store1.domain().shape();
  std::cout<<store1.dtype()<<std::endl;
  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto base_zarr_dtype,
                                     ChooseBaseDType(store1.dtype()));
  std::cout<<base_zarr_dtype.encoded_dtype<<std::endl;
  auto cur_x_max = static_cast<int>(shape[3]);
  auto cur_y_max = static_cast<int>(shape[4]);
  auto max_scale = static_cast<int>(ceil(log2(std::max({cur_x_max, cur_y_max}))));
  std::string base_zarr_file = output_file+"/" + std::to_string(max_scale)+"/";
  auto num_rows = static_cast<int>(ceil(1.0*cur_x_max/chunk_size));
  auto num_cols = static_cast<int>(ceil(1.0*cur_y_max/chunk_size));
  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store2, tensorstore::Open({{"driver", "zarr"},
                            {"kvstore", {{"driver", "file"},
                                         {"path", base_zarr_file}}
                            },
                            {"context", {
                              {"cache_pool", {{"total_bytes_limit", 1000000000}}},
                              {"data_copy_concurrency", {{"limit", 4}}},
                              {"file_io_concurrency", {{"limit", 4}}},
                            }},
                            {"metadata", {
                                          {"zarr_format", 2},
                                          {"shape", {cur_x_max, cur_y_max}},
                                          {"chunks", {chunk_size, chunk_size}},
                                          {"dtype", base_zarr_dtype.encoded_dtype},
                                          },
                            }},
                            //context,
                            tensorstore::OpenMode::create |
                            tensorstore::OpenMode::delete_existing,
                            tensorstore::ReadWriteMode::write).result());


  //std::list<std::future<double>> pending_writes;

  for(int i=0; i<num_rows; ++i){
    auto x_start = i*chunk_size;
    auto x_end = std::min({(i+1)*chunk_size, cur_x_max});
    for(int j=0; j<num_cols; ++j){
      auto y_start = j*chunk_size;
      auto y_end = std::min({(j+1)*chunk_size, cur_y_max});
      pool.submit([store1, store2, x_start, x_end, y_start, y_end](){  
                              //auto time1 = std::chrono::steady_clock::now();
                              auto array = tensorstore::AllocateArray({x_end-x_start, y_end-y_start},tensorstore::c_order,
                                                            tensorstore::value_init, store1.dtype());
                              // initiate a read
                              tensorstore::Read(store1 | 
                                        tensorstore::Dims(0).ClosedInterval(0,0) |
                                        tensorstore::Dims(1).ClosedInterval(0,0) |
                                        tensorstore::Dims(2).ClosedInterval(0,0) |
                                        tensorstore::Dims(3).ClosedInterval(x_start,x_end-1) |
                                        tensorstore::Dims(4).ClosedInterval(y_start,y_end-1) ,
                                        array).value();
                                                
                              // initiate write
                              tensorstore::Write(array, store2 |
                                  tensorstore::Dims(0).ClosedInterval(x_start,x_end-1) |
                                  tensorstore::Dims(1).ClosedInterval(y_start,y_end-1)).value();     
                              // wait for write

                              // auto time2 = std::chrono::steady_clock::now();
                              // std::chrono::duration<double> elapsed_seconds =  time2-time1;

                              // return elapsed_seconds.count();

      });       

      //pending_writes.emplace_back(std::async(CopyChunk<decltype(store1)>, store1, store2, x_start, x_end, y_start, y_end)); 
    }
  }
  pool.wait_for_tasks();
  // auto total_writes {pending_writes.size()};
  // //std::cout<< "total writes "<< pending_writes.size() <<std::endl;
  // while(total_writes > 0){
  //   for(auto &f: pending_writes){
  //     if(f.valid()){
  //       auto status = f.wait_for(10ms);
  //       if (status == std::future_status::ready){
  //         auto tmp = f.get();
  //         --total_writes;
  //       }
  //     }
  //   }
  // }
}

void write_downsampled_image(std::string& input_file, std::string& output_file){
  int chunk_size = 1024;
  tensorstore::Context context = Context::Default();
  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store1, tensorstore::Open({{"driver", "zarr"},
                          {"kvstore", {{"driver", "file"},
                                        {"path", input_file}}
                          },
                          {"context", {
                            {"cache_pool", {{"total_bytes_limit", 1000000000}}},
                            {"data_copy_concurrency", {{"limit", 4}}},
                            {"file_io_concurrency", {{"limit", 4}}},
                          }}},
                          //context,
                          tensorstore::OpenMode::open,
                          tensorstore::ReadWriteMode::read).result());
  auto shape = store1.domain().shape();
  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto base_zarr_dtype,
                                     ChooseBaseDType(store1.dtype()));
  auto prev_x_max = static_cast<int>(shape[0]);
  auto prev_y_max = static_cast<int>(shape[1]);

  auto cur_x_max = static_cast<int>(ceil(prev_x_max/2.0));
  auto cur_y_max = static_cast<int>(ceil(prev_y_max/2.0));

  auto num_rows = static_cast<int>(ceil(1.0*cur_x_max/chunk_size));
  auto num_cols = static_cast<int>(ceil(1.0*cur_y_max/chunk_size));

  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store2, tensorstore::Open({{"driver", "zarr"},
                          {"kvstore", {{"driver", "file"},
                                        {"path", output_file}}
                          },
                          {"context", {
                            {"cache_pool", {{"total_bytes_limit", 1000000000}}},
                            {"data_copy_concurrency", {{"limit", 4}}},
                            {"file_io_concurrency", {{"limit", 4}}},
                          }},
                          {"metadata", {
                                        {"zarr_format", 2},
                                        {"shape", {cur_x_max, cur_y_max}},
                                        {"chunks", {chunk_size, chunk_size}},
                                        {"dtype", base_zarr_dtype.encoded_dtype},
                                        },
                          }},
                          //context,
                          tensorstore::OpenMode::create |
                          tensorstore::OpenMode::delete_existing,
                          tensorstore::ReadWriteMode::write).result());

  std::list<std::future<double>> pending_writes;
  BS::thread_pool pool(4);
  for(int i=0; i<num_rows; ++i){
    auto x_start = i*chunk_size;
    auto x_end = std::min({(i+1)*chunk_size, cur_x_max});

    auto prev_x_start = 2*x_start;
    auto prev_x_end = std::min({2*x_end, prev_x_max});
    for(int j=0; j<num_cols; ++j){
      auto y_start = j*chunk_size;
      auto y_end = std::min({(j+1)*chunk_size, cur_y_max});
      auto prev_y_start = 2*y_start;
      auto prev_y_end = std::min({2*y_end, prev_y_max});
      //pending_writes.emplace_back(std::async(DownsampleAndWrtieChunk<decltype(store1), uint16_t>, store1, store2, x_start, x_end, y_start, y_end,
      //                                                                                              prev_x_start, prev_x_end, prev_y_start, prev_y_end)); 
      pool.submit([store1, store2, x_start, x_end, y_start, y_end, prev_x_start, prev_x_end, prev_y_start, prev_y_end](){  
                    std::vector<uint16_t> read_buffer((prev_x_end-prev_x_start)*(prev_y_end-prev_y_start));
                    auto array = tensorstore::Array(read_buffer.data(), {prev_x_end-prev_x_start, prev_y_end-prev_y_start}, tensorstore::c_order);

                    tensorstore::Read(store1 | 
                            tensorstore::Dims(0).ClosedInterval(prev_x_start,prev_x_end-1) |
                            tensorstore::Dims(1).ClosedInterval(prev_y_start,prev_y_end-1) ,
                            tensorstore::UnownedToShared(array)).value();

                    auto resutl = downsample_average(read_buffer, (prev_x_end-prev_x_start), (prev_y_end-prev_y_start));
                    auto result_array = tensorstore::Array(resutl->data(), {x_end-x_start, y_end-y_start}, tensorstore::c_order);
                    tensorstore::Write(tensorstore::UnownedToShared(result_array), store2 |
                        tensorstore::Dims(0).ClosedInterval(x_start,x_end-1) |
                        tensorstore::Dims(1).ClosedInterval(y_start,y_end-1)).value();     
                    // wait for write
                    //return 0.0;

      }); 
    }
  }
  // auto total_writes {pending_writes.size()};
  // //std::cout<< "total writes "<< pending_writes.size() <<std::endl;
  // while(total_writes > 0){
  //   for(auto &f: pending_writes){
  //     if(f.valid()){
  //       auto status = f.wait_for(10ms);
  //       if (status == std::future_status::ready){
  //         auto tmp = f.get();
  //         --total_writes;
  //       }
  //     }
  //   }
  // }
  pool.wait_for_tasks();
}


void read_ometiff_data()
{
  BS::thread_pool pool(8);
  auto time1 = std::chrono::steady_clock::now();
  tensorstore::Context context = Context::Default();
  std::cout<<"here" <<std::endl;
  std::string path = "/mnt/hdd8/axle/data/tmp_dir/input/p03_x(01-24)_y(01-16)_wx(0-2)_wy(0-2)_c1.ome.tif";
  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store1, tensorstore::Open({{"driver", "ometiff"},

                            {"kvstore", {{"driver", "tiled_tiff"},
                                         {"path", path}}
                            },
                            {"context", {
                              {"cache_pool", {{"total_bytes_limit", 1000000000}}},
                              {"data_copy_concurrency", {{"limit", 4}}},
                              {"file_io_concurrency", {{"limit", 4}}},
                            }}
                            },
                            //context,
                            tensorstore::OpenMode::open,
                            //tensorstore::RecheckCached{true},
                            //tensorstore::RecheckCachedData{false},
                            tensorstore::ReadWriteMode::read).result());

  std::cout<<"here" <<std::endl;
  auto shape = store1.domain().shape();
  std::cout<<shape<<std::endl;
  auto cur_x_max = static_cast<int>(shape[3]);
  auto cur_y_max = static_cast<int>(shape[4]);

  auto num_rows = static_cast<int>(ceil(1.0*cur_x_max/1024));
  auto num_cols = static_cast<int>(ceil(1.0*cur_y_max/1024));

  //int num_rows = 25;
  //int num_cols = 25;  

  //std::list<tensorstore::WriteFutures> pending_writes;
  std::list<std::future<double>> pending_writes;

  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store2, tensorstore::Open({{"driver", "zarr"},
                            {"kvstore", {{"driver", "file"},
                                         {"path", "/mnt/hdd8/axle/data/tmp_dir/test_zarr"}}
                            },
                            {"context", {
                              {"cache_pool", {{"total_bytes_limit", 1000000000}}},
                              {"data_copy_concurrency", {{"limit", 4}}},
                              {"file_io_concurrency", {{"limit", 4}}},
                            }}
                            ,
                            {"metadata", {
                                          {"zarr_format", 2},
                                          {"shape", {52910, 79390}},
                                          {"chunks", {1024, 1024}},
                                          {"dtype", "<u2"},
                                          },
                            }},
                            //context,
                            tensorstore::OpenMode::create |
                            tensorstore::OpenMode::delete_existing,
                            //tensorstore::RecheckCached{true},
                            //tensorstore::RecheckCachedData{false},
                            tensorstore::ReadWriteMode::write).result());
  auto time2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds =  time2-time1;
  std::cout<< "time taken to open" << elapsed_seconds.count()<<std::endl;
  for(int i=0; i<num_rows; ++i){
    auto x_start = i*1024;
    auto x_end = std::min({(i+1)*1024, cur_x_max});

    for(int j=0; j<num_cols; ++j){
      auto y_start = j*1024;
      auto y_end = std::min({(j+1)*1024, cur_y_max});
      //auto tmp = CopyChunk(store1, store2, x_start, x_end, y_start, y_end);
      //pending_writes.emplace_back(std::async(CopyChunk<decltype(store1)>, store1, store2, x_start, x_end, y_start, y_end));    
      //auto tmp = pool.submit(CopyChunk3<decltype(store1)>, &store1, &store2, x_start, x_end, y_start, y_end);   
      auto tmp = pool.submit([store1, store2, x_start, x_end, y_start, y_end](){  
                                    auto time1 = std::chrono::steady_clock::now();
                                    auto array = tensorstore::AllocateArray({x_end-x_start, y_end-y_start},tensorstore::c_order,
                                                                  tensorstore::value_init, store1.dtype());
                                    // initiate a read
                                    tensorstore::Read(store1 | 
                                              tensorstore::Dims(0).ClosedInterval(0,0) |
                                              tensorstore::Dims(1).ClosedInterval(0,0) |
                                              tensorstore::Dims(2).ClosedInterval(0,0) |
                                              tensorstore::Dims(3).ClosedInterval(x_start,x_end-1) |
                                              tensorstore::Dims(4).ClosedInterval(y_start,y_end-1) ,
                                              array).value();
                                                      
                                    // initiate write
                                    tensorstore::Write(array, store2 |
                                        tensorstore::Dims(0).ClosedInterval(x_start,x_end-1) |
                                        tensorstore::Dims(1).ClosedInterval(y_start,y_end-1)).value();     
                                    // wait for write

                                    auto time2 = std::chrono::steady_clock::now();
                                    std::chrono::duration<double> elapsed_seconds =  time2-time1;

                                    return elapsed_seconds.count();

      });       
      //pending_writes.emplace_back(pool.submit(CopyChunk2<decltype(store1)>, store1, store2, x_start, x_end, y_start, y_end));
    }
  }

  std::cout<< "total writes "<< pending_writes.size() <<std::endl;
  bool all_done = false;
  double elapsed_time{0.0};
  auto loop_count{0};
  auto total_writes {pending_writes.size()};
  std::vector<double> times;
  pool.wait_for_tasks();
  // while(total_writes > 0){
  //   ++loop_count;
  //   for(auto &f: pending_writes){
  //     if(f.valid()){
  //       auto status = f.wait_for(10ms);
  //       if (status == std::future_status::ready){
  //         auto t = f.get();
  //         elapsed_time += t;
  //         times.emplace_back(t);
  //         --total_writes;
  //       }
  //     }
  //   }
  // }
  // for(auto &f: pending_writes){
  //   auto t = f.get();
  //   elapsed_time += t;
  //   times.emplace_back(t);
  // }

  // for(auto& t: times){
  //   std::cout << "tile time " << t << std::endl;
  // }
  //std::cout<<"looped " << loop_count << std::endl;
  std::cout<<"total_elapsed_time in Copy " << elapsed_time << std::endl;

  //sleep(60);
}

void create_image_pyramids(){
  std::string base_path = "/mnt/hdd8/axle/data/tmp_dir/test_zarr/";
  for (int i=16; i>8; --i){
    std::string input_path = base_path + std::to_string(i+1) + "/";
    std::string output_path = base_path + std::to_string(i) + "/";
    auto time1 = std::chrono::steady_clock::now();
    write_downsampled_image(input_path, output_path);
    auto time2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds =  time2-time1;
    std::cout<< "time taken at scale "<< i <<":" << elapsed_seconds.count()<<std::endl;
  }
}

void test_create_image_pyramids(){
  create_image_pyramids();
}

void test_write_downsampled_image()
{
  std::string input_file = "/mnt/hdd8/axle/data/tmp_dir/test_zarr/17/";
  std::string output_file = "/mnt/hdd8/axle/data/tmp_dir/test_zarr/16/";
  write_downsampled_image(input_file, output_file);
}

void test_write_base_scale()
{
  std::string input_file = "/mnt/hdd8/axle/data/tmp_dir/input/p03_x(01-24)_y(01-16)_wx(0-2)_wy(0-2)_c1.ome.tif";
  std::string output_file = "/mnt/hdd8/axle/data/tmp_dir/test_zarr/";
  write_base_scale(input_file, output_file);
}

int main(int argc, char** argv) {
  auto time1 = std::chrono::steady_clock::now();
  //read_ometiff_data();
  test_write_base_scale();
  auto time2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds =  time2-time1;
  std::cout<< "time taken " << elapsed_seconds.count()<<std::endl;
  //test_write_downsampled_image();
  //sleep(5);
  test_create_image_pyramids();
  auto time3 = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds2 =  time3-time1;
  std::cout<< "time taken " << elapsed_seconds2.count()<<std::endl;
 return 0;
}