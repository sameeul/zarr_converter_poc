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

def _avg2(image: np.ndarray) -> np.ndarray:
    """ Average pixels together with optical field 2x2 and stride 2
    
    Args:
        image - numpy array with only two dimensions (m,n)
    Returns:
        avg_img - numpy array with only two dimensions (round(m/2),round(n/2))
    """
    
    # Since we are adding pixel values, we need to update the pixel type 
    # This helps to avoid integer overflow
    if image.dtype == np.uint8:
        dtype = np.uint16
    elif image.dtype == np.uint16:
        dtype = np.uint32
    elif image.dtype == np.uint32:
        dtype = np.uint64
    elif image.dtype == np.int8:
        dtype = np.int16
    elif image.dtype == np.int16:
        dtype = np.int32
    elif image.dtype == np.int32:
        dtype = np.int64
    else:
        dtype = image.dtype
        
    odtype = image.dtype
    image = image.astype(dtype)
    
    y_max = image.shape[0] - image.shape[0] % 2
    x_max = image.shape[1] - image.shape[1] % 2
    
    # Calculate the mean
    avg_img = np.zeros(np.ceil([d/2 for d in image.shape]).astype(int),dtype=dtype)
    avg_img[0:y_max//2,0:x_max//2] = (image[0:y_max-1:2, 0:x_max-1:2] + \
                                      image[1:  y_max:2, 0:x_max-1:2] + \
                                      image[0:y_max-1:2,   1:x_max:2] + \
                                      image[1:  y_max:2,   1:x_max:2]) // 4
    
    # Fill in the final row if the image height is odd-valued
    if y_max != image.shape[0]:
        avg_img[-1,:x_max//2] = (image[-1,0:x_max-1:2] + \
                                 image[-1,1:x_max:2]) // 2
    # Fill in the final column if the image width is odd-valued
    if x_max != image.shape[1]:
        avg_img[:y_max//2,-1] = (image[0:y_max-1:2,-1] + \
                                 image[1:y_max:2,-1]) // 2
    # Fill in the lower right pixel if both image width and height are odd
    if y_max != image.shape[0] and x_max != image.shape[1]:
        avg_img[-1,-1] = image[-1,-1]
        
    return avg_img.astype(odtype)

class ZarrStreamer:
    def __init__(self, input_file, output_file, dtype, levels, shape, physical_size, chunk_size) -> None:
        self._input_file = input_file
        self._output_file = output_file
        self._dtype = dtype
        self._pyramid_levels = levels
        self._base_shape = shape
        self._physical_size = physical_size
        self._chunk_size = chunk_size
        self._scale_keys = {} 
        self._zarr_multiscale_datasets = []
        self._max_scale = int(np.ceil(np.log2(max(self._base_shape))))
        self._min_scale = self._max_scale - self._pyramid_levels
        if self._min_scale <=0:
            self._min_scale = 1


    def create_metadata(self):

        # create the base layer
        zarr_schema_at_each_scale = {
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


        zarr_multiscale_datasets = [
            {
                "path" : f"{str(self._max_scale)}",
                "coordinateTransformations": [{
                        "type": "scale",
                        "scale": [1.0, 1.0, 1.0]
                    }]
            }
        ]

        zarr_multiscale_axes = [
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"}
            ]
        
        self._scale_keys[str(self._max_scale)] = ts.open(zarr_schema_at_each_scale).result()

        for i in reversed(range(self._min_scale, self._max_scale)):
            zarr_schema_at_each_scale['kvstore'] = f"file://{self._output_file}/{str(i)}"
            zarr_schema_at_each_scale['metadata']['shape'] = [1,int(np.ceil(zarr_schema_at_each_scale['metadata']['shape'][1]/2)),int(np.ceil(zarr_schema_at_each_scale['metadata']['shape'][2]/2))]
            self._scale_keys[str(i)] = ts.open(zarr_schema_at_each_scale).result()
            prev_dataset = zarr_multiscale_datasets[-1]
            curr_dataset = copy.deepcopy(prev_dataset)
            curr_dataset["path"] = str(i)
            curr_dataset["coordinateTransformations"][0]["scale"] = [  curr_dataset["coordinateTransformations"][0]["scale"][0], 
                                                                    curr_dataset["coordinateTransformations"][0]["scale"][1]*2,
                                                                    curr_dataset["coordinateTransformations"][0]["scale"][2]*2,
                                                                    ]
            zarr_multiscale_datasets.append(curr_dataset)

        zarr_multiscale_metdata = {
        "multiscales" :    [
                {
                    "name": os.path.basename(self._input_file),
                    "version":  "0.4",
                    "axes": zarr_multiscale_axes,
                    "datasets": zarr_multiscale_datasets,
                    "metadata": {
                        "method": "mean"
                        }
                }
            ]
        }

        with open(f"{self._output_file}/.zattrs", "w") as fp:
            json.dump(zarr_multiscale_metdata, fp)

    def write_base_scale(self):
        br = BioReader(self._input_file, 8)
        out_dataset = self._scale_keys[str(self._max_scale)]
        task_set = set()
        cur_x_max = out_dataset.shape[1]
        cur_y_max = out_dataset.shape[2]

        num_rows = int(np.ceil(cur_x_max/self._chunk_size))
        num_cols = int(np.ceil(cur_y_max/self._chunk_size))


        for i in range(num_rows):
            x_start = i*self._chunk_size
            x_end = min((i+1)*self._chunk_size, cur_x_max)

            for j in range(num_cols):
                y_start = j*self._chunk_size
                y_end = min((j+1)*self._chunk_size, cur_y_max)
                tmp = out_dataset[0,x_start:x_end, y_start:y_end].write(br._image_reader[0,0,0,x_start:x_end, y_start:y_end].read().result())
                #tmp = out_dataset[0,x_start:x_end, y_start:y_end].write(read_task[(x_start,x_end, y_start,y_end)].result())
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
        for i in reversed(range(self._min_scale, self._max_scale)):
            prev_dataset = self._scale_keys[str(i+1)]
            current_dataset = self._scale_keys[str(i)]

            prev_x_max = prev_dataset.shape[1]
            prev_y_max = prev_dataset.shape[2]

            cur_x_max = current_dataset.shape[1]
            cur_y_max = current_dataset.shape[2]

            num_rows = int(np.ceil(cur_x_max/self._chunk_size))
            num_cols = int(np.ceil(cur_y_max/self._chunk_size))

            task_set = set()
            for i in range(num_rows):
                x_start = i*self._chunk_size
                x_end = min((i+1)*self._chunk_size, cur_x_max)

                prev_x_start = 2*x_start
                prev_x_end = min(2*x_end, prev_x_max)
                for j in range(num_cols):
                    y_start = j*self._chunk_size
                    y_end = min((j+1)*self._chunk_size, cur_y_max)
                    prev_y_start = 2*y_start
                    prev_y_end = min(2*y_end, prev_y_max)
                    tmp = current_dataset[0,x_start:x_end, y_start:y_end].write(_avg2(prev_dataset[0,prev_x_start:prev_x_end, prev_y_start:prev_y_end].read().result()))
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

