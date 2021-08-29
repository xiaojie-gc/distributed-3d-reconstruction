#!/bin/bash     

if (($1 == 1))
then
    python3 main.py --data_dir data/originals --bg 1 --bg_new 0 --worker 2 --max_r 1280 --time 0 --n 7 --edge_fg [000]-[003] --local_fg [004]-[006]
elif (($1 == 2))
then
    python3 main.py --data_dir data/originals --bg 1 --bg_new 1 --worker 2 --max_r 960 --time 0 --n 7 --edge_fg [000]-[003] --local_fg [004]-[006] --edge_bg [000]-[003] --local_bg [004]-[006]
elif (($1 == 3))
then
    python3 main.py --data_dir /media/zxj/easystore/Dance1/data/originals --source_dir data/Family --profile 0 --new_cfg 0 --bg 0 --bg_new 0 --worker 1 --max_r 960  --time 0 --n 7 --local [000]-[006]
elif (($1 == 4))
then
    python3 main.py --data_dir /media/zxj/easystore/Dance1/data/originals --bg 1 --bg_new 0 --profile 0 --worker 1 --max_r 960  --time 0 --n 7 --local_fg [000]-[006] --local_bg [000]-[006]
elif (($1 == 5))
then
    python3 MvgMvsPipeline.py data/dslr_images_undistorted data/gold_results_960/00000_output --sfm sfm_data.bin --mvs_dir mvs
elif (($1 == 6))
then
     python3 MvgMvsPipeline.py data/collect/00000 data/gold_results_960/00000_output --sfm sfm_data.bin --mvs_dir mvs --preset MVG_MVS --tasks [000]-[006] --resolution 960
elif (($1 == 7))
then
    python3 MvgMvsPipeline.py data/collect/00000 data/gold_results_960/00000_output --sfm sfm_data.bin --mvs_dir mvs --preset DensifyPointCloud --tasks [000]-[006] --resolution 192 --do_fuse "1"
elif (($1 == 8))
then
    python3 gold_3d_remote.py --bg 0 --bg_new 0 --worker 1 --max_r 960  --time 0 --n 7 --local 0123456 --edge None
fi
