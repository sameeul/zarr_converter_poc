from neuglancer_streamer import NeuroglancerStreamer
from zarr_streamer import ZarrStreamer
from zarr_streamer_2 import ZarrStreamer as ZarrStreamer2
import time

def test_ds():
    print("Downsampling Code")
    input_file_name = '/mnt/hdd8/axle/data/bfio_test_images/r001_c001_z000.ome.tif'
    output_file_name = '/mnt/hdd8/axle/data/bfio_test_images/r001_c001_z000_zarr_2'

    start_time = time.time()
    ngs = ZarrStreamer( input_file_name,
                                output_file_name, 
                                "uint16", 
                                7, 
                                [29286, 42906, 1],
                                [(0.325, "µm"), (0.325, "µm"), (1, "µm")],
                                1024)
    
    ngs.write_base_scale()
    ngs.write_pyramid_scales()
    ngs.write_multiscale_metadata()

    end_time = time.time()
    print(end_time-start_time)

def test_avg():
    print("Our implementation")
    input_file_name = '/mnt/hdd8/axle/data/bfio_test_images/r001_c001_z000.ome.tif'
    output_file_name = '/mnt/hdd8/axle/data/bfio_test_images/r001_c001_z000_zarr_3'

    start_time = time.time()
    ngs = ZarrStreamer2( input_file_name,
                                output_file_name, 
                                "uint16", 
                                7, 
                                [29286, 42906, 1],
                                [(0.325, "µm"), (0.325, "µm"), (1, "µm")],
                                1024)
    
    ngs.create_metadata()
    ngs.write_base_scale()
    #ngs.write_pyramid_scales()

    end_time = time.time()
    print(end_time-start_time)

if __name__ == '__main__':
    #test_ds()
    test_avg()
