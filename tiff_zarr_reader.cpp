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
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; ++j) {
      int new_data_index = static_cast<int>(floor(i / 2.0)) * new_col +
                           static_cast<int>(floor(j / 2.0));
      data_ptr[new_data_index] =
          data_ptr[new_data_index] + 0.25 * source_array[i * col + j];
    }
  }
  // fix the last col if odd
  if (col%2 == 1){
    for(int j=0; j<new_col; ++j){
      data_ptr[(new_row-1)*new_col+j] *=2;
    }
  }

  // fix the last row if odd
  if (row%2 == 1){
    for(int i=0; i<new_row; ++i){
      data_ptr[i*new_col+(new_col-1)] *=2;
    }
  }
  return std::move(result_data);
}

template <typename T>
double DownsampleAndWrtieChunk(T&& source, T&& dest, int x1, int x2, int y1, int y2){
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
double CopyChunk3(T&& source, T&& dest, int x1, int x2, int y1, int y2)
{
  auto time1 = std::chrono::steady_clock::now();
  auto array = tensorstore::AllocateArray<tensorstore::uint16_t>({x2-x1, y2-y1});
  // initiate a read
  tensorstore::Read(source | 
            tensorstore::Dims(0).ClosedInterval(0,0) |
            tensorstore::Dims(1).ClosedInterval(x1,x2-1) |
            tensorstore::Dims(2).ClosedInterval(y1,y2-1) ,
            array).value();
  // wait for read 
  //while(source)  
  // initiate write
  tensorstore::Write(array, dest | 
      tensorstore::Dims(0).IndexSlice(0) |
      tensorstore::Dims(1).ClosedInterval(x1,x2-1) |
      tensorstore::Dims(2).ClosedInterval(y1,y2-1)).value();     
  // wait for write

  auto time2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds =  time2-time1;

  return elapsed_seconds.count();
}

void write_base_scale(std::string input_file, std::string output_file){

  int chunk_size = 1024;
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


  std::list<std::future<double>> pending_writes;

  for(int i=0; i<num_rows; ++i){
    auto x_start = i*chunk_size;
    auto x_end = std::min({(i+1)*chunk_size, cur_x_max});
    for(int j=0; j<num_cols; ++j){
      auto y_start = j*chunk_size;
      auto y_end = std::min({(j+1)*chunk_size, cur_y_max});
      pending_writes.emplace_back(std::async(CopyChunk<decltype(store1)>, store1, store2, x_start, x_end, y_start, y_end)); 
    }
  }
  auto total_writes {pending_writes.size()};
  //std::cout<< "total writes "<< pending_writes.size() <<std::endl;
  while(total_writes > 0){
    for(auto &f: pending_writes){
      if(f.valid()){
        auto status = f.wait_for(10ms);
        if (status == std::future_status::ready){
          auto tmp = f.get();
          --total_writes;
        }
      }
    }
  }
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
                          }},
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
      pending_writes.emplace_back(std::async(DownsampleAndWrtieChunk<decltype(store1)>, store1, store2, x_start, x_end, y_start, y_end,
                                                                                                    prev_x_start, prev_x_end, prev_y_start, prev_y_end)); 
    }
  }
  auto total_writes {pending_writes.size()};
  //std::cout<< "total writes "<< pending_writes.size() <<std::endl;
  while(total_writes > 0){
    for(auto &f: pending_writes){
      if(f.valid()){
        auto status = f.wait_for(10ms);
        if (status == std::future_status::ready){
          auto tmp = f.get();
          --total_writes;
        }
      }
    }
  }
}


void read_ometiff_data()
{
  BS::thread_pool pool(4);
  auto time1 = std::chrono::steady_clock::now();
  tensorstore::Context context = Context::Default();
  std::cout<<"here" <<std::endl;
  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store1, tensorstore::Open({{"driver", "ometiff"},

                            {"kvstore", {{"driver", "tiled_tiff"},
                                         {"path", "/mnt/hdd8/axle/data/bfio_test_images/r001_c001_z000.ome.tif"}}
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
                                         {"path", "/mnt/hdd8/axle/data/bfio_test_images/r001_c001_z000_zarr_2/18/"}}
                            },
                            {"context", {
                              {"cache_pool", {{"total_bytes_limit", 1000000000}}},
                              {"data_copy_concurrency", {{"limit", 4}}},
                              {"file_io_concurrency", {{"limit", 4}}},
                            }}
                            ,
                            {"metadata", {
                                          {"zarr_format", 2},
                                          {"shape", {29286, 42906}},
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
      pending_writes.emplace_back(std::async(CopyChunk<decltype(store1)>, store1, store2, x_start, x_end, y_start, y_end));    
      //auto tmp = pool.submit(CopyChunk2<decltype(store1)>, store1, store2, x_start, x_end, y_start, y_end);          
      //pending_writes.emplace_back(pool.submit(CopyChunk2<decltype(store1)>, store1, store2, x_start, x_end, y_start, y_end));
    }
  }

  std::cout<< "total writes "<< pending_writes.size() <<std::endl;
  bool all_done = false;
  double elapsed_time{0.0};
  auto loop_count{0};
  auto total_writes {pending_writes.size()};
  std::vector<double> times;
//  pool.wait_for_tasks();
  while(total_writes > 0){
    ++loop_count;
    for(auto &f: pending_writes){
      if(f.valid()){
        auto status = f.wait_for(10ms);
        if (status == std::future_status::ready){
          auto t = f.get();
          elapsed_time += t;
          times.emplace_back(t);
          --total_writes;
        }
      }
    }
  }
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



void test_write_base_scale()
{
  std::string input_file = "/mnt/hdd8/axle/data/bfio_test_images/r001_c001_z000.ome.tif";
  std::string output_file = "/mnt/hdd8/axle/data/bfio_test_images/r001_c001_z000_zarr_2";
  write_base_scale(input_file, output_file);
}

int main(int argc, char** argv) {
  auto time1 = std::chrono::steady_clock::now();
  //read_ometiff_data();
  
  test_write_base_scale();
  auto time2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds =  time2-time1;
  std::cout<< "time taken " << elapsed_seconds.count()<<std::endl;
 return 0;
}