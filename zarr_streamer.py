import tensorstore as ts
import numpy as np
from bioreader import BioReader
import copy
import time
import os
import json

UNITS = {'m':  10**9,
         'cm': 10**7,
         'mm': 10**6,
         'µm': 10**3,
         'nm': 1,
         'Å':  10**-1,
         "": 1}

class ZarrStreamer:
    def __init__(self, input_file, output_file, dtype, levels, shape, physical_size, chunk_size) -> None:
        self._input_file = input_file
        self._output_file = output_file
        self._dtype = dtype
        self._pyramid_levels = levels
        self._base_shape = shape
        self._physical_size = physical_size
        self._chunk_size = chunk_size
        self._zarr_multiscale_datasets = []
        self._max_scale = int(np.ceil(np.log2(max(self._base_shape))))
        self._min_scale = self._max_scale - self._pyramid_levels
        if self._min_scale <=0:
            self._min_scale = 1

    def write_multiscale_metadata(self):
        
        zarr_multiscale_axes = [
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"}
            ]
        
        zarr_multiscale_metdata = {
        "multiscales" :    [
                {
                    "name": os.path.basename(self._input_file),
                    "version":  "0.4",
                    "axes": zarr_multiscale_axes,
                    "datasets": self._zarr_multiscale_datasets,
                    "metadata": {
                        "method": "mean"
                        }
                }
            ]
        }
        with open(f"{self._output_file}/.zattrs", "w") as fp:
            json.dump(zarr_multiscale_metdata, fp)  



    def write_base_scale(self):
        self._zarr_multiscale_datasets.append(
            {
                "path" : f"{str(self._max_scale)}",
                "coordinateTransformations": [{
                        "type": "scale",
                        "scale": [1.0, 1.0, 1.0]
                    }]
            }
        )
        zarr_ts_future = ts.open({
                'driver': 'zarr',
                'kvstore':  f"file://{self._output_file}/{str(self._max_scale)}",
                'metadata': {
                    'zarr_format' : 2,
                    'shape' : [self._base_shape[2], self._base_shape[0], self._base_shape[1]],
                    'chunks' : [1, self._chunk_size, self._chunk_size],
                    'dtype' : "<u2",
                },
                'create': True,
                'delete_existing': True,
                }
                )
            
        zarr_ts = zarr_ts_future.result()
        cur_x_max = zarr_ts.shape[1]
        cur_y_max = zarr_ts.shape[2]
        task_set = set()
        num_rows = int(np.ceil(cur_x_max/self._chunk_size))
        num_cols = int(np.ceil(cur_y_max/self._chunk_size))
        br = BioReader(self._input_file, 8)
        for i in range(num_rows):
            x_start = i*self._chunk_size
            x_end = min((i+1)*self._chunk_size, cur_x_max)

            for j in range(num_cols):
                y_start = j*self._chunk_size
                y_end = min((j+1)*self._chunk_size, cur_y_max)
                tmp = zarr_ts[0,x_start:x_end, y_start:y_end].write(br._image_reader[0,0,0,x_start:x_end, y_start:y_end].read().result())
                task_set.add(tmp)



        # loop till all the WritePromises are done
        # will look into a better way for this later
        all_done = False
        while not all_done:
            task_done = 0
            for task in task_set:
                if task.done() != True:
                    break
                else:
                    task_done += 1

            if task_done == len(task_set):
                all_done = True   



    def write_pyramid_scales(self):
        factor = 1
        for i in reversed(range(self._min_scale, self._max_scale)):
            factor = 2*factor
            downsample_ts_future = ts.open({
                    "driver": "downsample",
                    "downsample_factors": [1, factor, factor],
                    "downsample_method": "mean",
                    "base": { 
                                'driver'    : 'zarr',
                                'kvstore'   : { 
                                    'driver' : 'file',
                                    'path' : f"{self._output_file}/{str(self._max_scale)}/",
                                }
                            }
                    }
                    )
            downsameple_ts = downsample_ts_future.result()

            zarr_ts_future = ts.open({
                'driver': 'zarr',
                'kvstore':  f"file://{self._output_file}/{str(i)}",
                'metadata': {
                    'zarr_format' : 2,
                    'shape' : downsameple_ts.shape,
                    'chunks' : [1, self._chunk_size, self._chunk_size],
                    'dtype' : "<u2",
                    },
                'create': True,
                'delete_existing': True,
                }
                )
            
            zarr_ts = zarr_ts_future.result()
            self._zarr_multiscale_datasets.append(
                {
                    "path" : f"{str(i)}",
                    "coordinateTransformations": [{
                            "type": "scale",
                            "scale": [1, factor, factor]
                        }]
                }
            )


            task_set = set()

            cur_x_max = zarr_ts.shape[1]
            cur_y_max = zarr_ts.shape[2]

            num_rows = int(np.ceil(cur_x_max/self._chunk_size))
            num_cols = int(np.ceil(cur_y_max/self._chunk_size))
            for i in range(num_rows):
                x_start = i*self._chunk_size
                x_end = min((i+1)*self._chunk_size, cur_x_max)

                for j in range(num_cols):
                    y_start = j*self._chunk_size
                    y_end = min((j+1)*self._chunk_size, cur_y_max)
                    tmp = zarr_ts[0,x_start:x_end, y_start:y_end].write(downsameple_ts[0,x_start:x_end, y_start:y_end].read().result())
                    task_set.add(tmp)

                all_done = False
                while not all_done:
                    task_done = 0
                    for task in task_set:
                        if task.done() != True:
                            break
                        else:
                            task_done += 1
                    if task_done == len(task_set):
                        all_done = True   
