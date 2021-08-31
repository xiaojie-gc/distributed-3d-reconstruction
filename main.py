import random
from _thread import start_new_thread
from threading import Lock
import argparse
from common.components import *
import psutil
import numpy as np

from common.networking import user_discovery, server_discovery
from detector import Server, detect
from common.DTU import prepare_img_dir
from common.fit_model import fit_return, objective
from initial import *
from main_bg import *

parser = argparse.ArgumentParser(description='Example command:')
args = initial_parameters(parser)
logger_parameters(args)
task_setting = initial_task_setting(args, task_setting_file="task.json")


######################################################################
"""
    For new datasets: data/Horse 00001.jpg, 00002.jpg, 00003.jpg
                      to
                      data/DTU_Horse 001/00001.jpg 002/00001.jpg 003/00001.jgp
"""
if args.new_cfg == "1":
    prepare_img_dir(args.data_dir, args.source_dir, int(args.n))
    print(logger() + f"+ initial data structure for {bcolors.OKGREEN}{args.source_dir}{bcolors.ENDC}")
########################################################################
m = Lock()

cpus = []


def CPU():
    while True:
        m.acquire()
        cpus.append(round(psutil.cpu_percent(0.1), 1))
        m.release()
        time.sleep(0.1)


cameras = [str(i).zfill(3) for i in range(int(args.n))]
dirs = cameras
print(logger() + "task_list", dirs)
##########################################################################################################
"""
    If images are sent from a remote user.
"""
if args.remote == "yes":
    user_channel = user_discovery(port=8003)
##########################################################################################################
"""
    Establish server connection.
"""
server_channel = []
if args.worker == 2:
    server_channel = server_discovery(port=8001)
##########################################################################################################

start_new_thread(CPU, ())

timestamp = args.time

# [192, 240, 288, 336, 384, 432, 480, 576, 672, 768, 864, 960]

all_r = ["192", "240", "288", "336", "384", "432", "480", "576", "672", "768", "864", "960"]

inx = 0

pred_error = {
    "local": {
        "fg_mvs": [],
        "bg_mvs": [],
        "mvs": []
    },
    "edge": {
        "fg_mvs": [],
        "bg_mvs": [],
        "mvs": []
    }

}

cp_time = {
    "local": {
        "fg_mvs": [],
        "bg_mvs": [],
        "mvs": []
    },
    "edge": {
        "fg_mvs": [],
        "bg_mvs": [],
        "mvs": []
    }
}

a, b, c, static_overhead = [], [], [], []
view_complexity = []

for i in range(args.worker):
    a.append(0)
    b.append(0)
    c.append(0)
    static_overhead.append({
        "mvs": 0.45,
        "fg_mvs": 0.5,
        "bg_mvs": 0.5
    })

ba, bb, bc = 0, 0, 0
b_ratio, b_source = 0, 5

if args.bg == 0:
    obs_profile = {
        "192": 0,
        "240": 0,
        "384": 0,
        "768": 0
    }
    subNet = None
else:
    """
    obs_profile = {
        "0.18": {
            "960": 0,
            "460": 0
        },
        "0.3" : {}        
    }
    """
    obs_profile = {}
    subNet = Server(args)

source_images = {
    "fg_mvs": 5,
    "bg_mvs": 5,
    "mvs": 5
}

while True:
    args.max_r = all_r[inx] if 2 == 2 else all_r[random.randint(4, 8)]
    str_timestamp, collect_dir, output_dir, fg_dir, bg_dir, result_dir = create_new_dir(args, timestamp)
    ####################################################################################################
    # SfM pipeline starts from here
    start = time.time()
    if args.remote == "no":
        network_delay, str_log, dirs = local_dataloader(str_timestamp, args.resolution, args.compress, args.data_dir,
                                                        collect_dir, dirs)
    else:
        network_delay = remote_dataloader(user_channel, collect_dir)
    m.acquire()
    print(logger(str_timestamp) + f"{bcolors.OKCYAN}current timestamp:{str_timestamp}{bcolors.ENDC}",
          f'{bcolors.WARNING}#{bcolors.ENDC}' * 75)
    print(logger(str_timestamp) + f"+ {bcolors.OKGREEN}-{args.max_r} image loading{bcolors.ENDC} in", network_delay,
          f" {bcolors.HEADER}[CPU:{round(np.average(cpus), 1)}%]{bcolors.ENDC}", str_log)
    cpus = []
    m.release()
    ####################################################################################################
    # load and update global sfm_data.json
    if args.new_cfg != "1":
        update_sfm_file(args, collect_dir, str_timestamp, DEL_list=[])
    ######################################################################################
    # SfM pipeline
    time_openMVG = openMVG(args.new_cfg, args.reconstructor, collect_dir, args.output_dir, str_timestamp)
    m.acquire()
    print(logger(str_timestamp) + f"+ {bcolors.OKGREEN}openMVG{bcolors.ENDC} in", time_openMVG,
          f"{bcolors.HEADER}[CPU:{round(np.average(cpus), 1)}%]{bcolors.ENDC}")
    cpus = []
    m.release()
    ####################################################################################################
    # create new sfm_data.json for a new dataset at the first time
    if args.new_cfg == "1":
        create_new_dataset_cfg(output_dir, args.source_dir, args.n)
    ####################################################################################################
    # application profiling with small resolution
    if args.profile == 1:
        if args.bg == 0:
            opeMVG2openMVS(args.reconstructor, collect_dir, output_dir, "sfm_data.bin", "mvs")
            """
            a_, b_, c_, obs_profile = profile(str_timestamp, args.reconstructor, collect_dir, output_dir, obs_profile)
            for i in range(args.worker):
                a[i] = a_
                b[i] = b_
                if i > 0:
                    b[i] = b[i] * 0.8
                c[i] = c_
            """
            a[0] = -0.000195431179417
            b[0] = 0.000003136040791
            c[0] = 0.022574519676297

            view_complexity = [
                [0.9762446431405747, 0.9636527167874716, 0.9714300830643882, 1.0206867361515264, 0.955875350510555, 1.032537960954447, 1.0795725093910375],
                #[0.9487222245382916, 0.96331346145829, 0.9679826572726895, 1.0712886146662777, 0.9539750698294912, 1.0222620586150832, 1.0724559136198775]
                [0.9762446431405747, 0.9636527167874716, 0.9714300830643882, 1.0206867361515264, 0.955875350510555, 1.032537960954447, 1.0795725093910375]
            ]

            if args.worker == 2:
                """
                a[1] = -0.000189220190737
                b[1] = 0.000003922930428
                c[1] = 0.018528047282126
                """
                a[1] = a[0] #  -0.000189220190737
                b[1] = b[0] / 0.8 # 0.000003922930428
                c[1] = c[0] # 0.018528047282126

            """
            for i in range(1, args.worker):
                a[i] = a[0]
                b[i] = b[0] / 0.8
                c[i] = a[0]
            """

        else:
            obs_profile = {
                "0.11": {
                    "192": 0.19099999999999998,
                    "240": 0.272,
                    "288": 0.365,
                    "336": 0.47400000000000003,
                    "384": 0.6139999999999999,
                    "432": 0.753,
                    "480": 0.92,
                    "576": 1.297,
                    "672": 1.7219999999999998,
                    "768": 2.2159999999999997,
                    "864": 2.765,
                    "960": 3.3760000000000003
                },
                "0.19": {
                    "192": 0.29200000000000004,
                    "240": 0.44100000000000006,
                    "336": 0.7600000000000001,
                    "432": 1.1669999999999998,
                    "576": 2.013,
                    "864": 4.28
                },
                "0.28": {
                    "192": 0.371,
                    "240": 0.562,
                    "336": 0.9960000000000001,
                    "432": 1.569,
                    "576": 2.711,
                    "864": 5.913000000000001
                },
                "0.38": {
                    "192": 0.525,
                    "240": 0.741,
                    "336": 1.3559999999999999,
                    "432": 2.158,
                    "576": 3.6929999999999996,
                    "864": 7.991
                }
            }
            ratio, obs_profile, ba, bb, bc, b_source = profile_bg(str_timestamp, args.reconstructor,
                                                                             args.bg_new, dirs, collect_dir, output_dir,
                                                                             fg_dir, bg_dir,
                                                                             subNet, obs_profile, profile=False)
            a, b, c = fit_profile(str_timestamp, obs_profile, a, b, c)

        args.profile = 0
    ####################################################################################################
    time_merge = 0
    if args.bg == 1:
        time_densify, a, b, c, obs_profile = bg(args, str_timestamp, a, b, c, args.bg_new, args.reconstructor, dirs, collect_dir, fg_dir, bg_dir, output_dir,
            subNet, b_ratio, ba, bb, bc, source_images, b_source, static_overhead, cp_time, pred_error, task_setting,
            server_channel, view_complexity, obs_profile)
        ######################################################################################
        time_fuse = fuse(args.max_r, args.reconstructor, collect_dir, output_dir, "fg_mvs", str_timestamp)
        if args.bg_new == 1:
            time_fuse += fuse(args.max_r, args.reconstructor, collect_dir, output_dir, "bg_mvs", str_timestamp)
        m.acquire()
        print(logger(str_timestamp) + f"+ {bcolors.OKGREEN}fuse{bcolors.ENDC} in", round(time_fuse, 4),
              f"{bcolors.HEADER}[CPU:{round(np.average(cpus), 1)}%]{bcolors.ENDC}")
        cpus = []
        m.release()
        ######################################################################################
        if args.bg_new == 0:
            ttt = "00002"
            # str_timestamp
            bg_path = "data/bg_scene_dense/" + ttt + "_scene_dense.ply"
        else:
            bg_path = output_dir + "/bg_mvs/scene_dense.ply"
        time_merge = do_merge(fg_path=output_dir + "/fg_mvs/scene_dense.ply",
                              bg_path=bg_path,
                              output_path="data/" + result_dir + "/" + str_timestamp + "_scene_dense.ply")
        m.acquire()
        print(logger(str_timestamp) + f"+ {bcolors.OKBLUE}merge{bcolors.ENDC} in", time_merge)
        cpus = []
        m.release()
    else:
        ######################################################################################
        opeMVG2openMVS(args.reconstructor, collect_dir, output_dir, "sfm_data.bin", "mvs")
        ######################################################################################
        # estimate computation latency
        time_densify, tag, static_overhead, source_images, depth_map_time = DensifyPointCloud_task(
            args.worker,
            args.bg, args.bg_new, cp_time,
            pred_error, output_dir, args.n,
            args.max_r, str_timestamp,
            task_setting, server_channel,
            source_images,
            view_complexity, static_overhead, a, b, c)

        #if args.max_r not in obs_profile:
        #    obs_profile[args.max_r] = depth_map_time["mvs"][0]
        #else:
        #    obs_profile[args.max_r] = obs_profile[args.max_r] * 0.3 + 0.7 * depth_map_time["mvs"][0]

        #a[0], b[0], c[0], obs_profile = profile(str_timestamp, args.reconstructor, collect_dir, output_dir, obs_profile, profile=False)

        #for i in range(1, args.worker):
        #    b[i] = b[i] * 0.3 + b[0] / (depth_map_time["mvs"][0] / depth_map_time["mvs"][i]) * 0.7

        m.acquire()
        print(
            logger(str_timestamp) + f"+ {bcolors.OKGREEN}Densify{bcolors.ENDC}:{bcolors.WARNING}{bcolors.ENDC} in",
            time_densify["total"],
            f"{bcolors.HEADER}[CPU:{round(np.average(cpus),1)}%]{bcolors.ENDC}")
        cpus = []
        m.release()
        ######################################################################################
        time_fuse = fuse(args.max_r, args.reconstructor, collect_dir, output_dir, "mvs", str_timestamp)
        m.acquire()
        print(logger(str_timestamp)  + f"+ {bcolors.OKGREEN}fuse{bcolors.ENDC} in", time_fuse,
              f"{bcolors.HEADER}[CPU:{round(np.average(cpus),1)}%]{bcolors.ENDC}")
        cpus = []
        m.release()
        shutil.copy2(os.path.join(output_dir, "mvs", "scene_dense.ply"), "data/"+result_dir+"/" + str_timestamp + "_scene_dense.ply")
        ######################################################################################
    time_total = time.time() - start

    print(logger(str_timestamp) + f"+ {bcolors.OKGREEN}evaluation{bcolors.ENDC}......")
    # precisions, recalls, fscores = eva(str_timestamp, output_path="data/"+result_dir+"/" + str_timestamp + "_scene_dense.ply", truth_path="data/no_background_sub_960_2/00000_scene_dense.ply")

    save_time(str_timestamp, args.compress, time_total, network_delay, time_openMVG, time_densify, time_fuse,
              None,
              None, None, time_merge)

    # timestamp = random.randint(0, 200)

    inx += 1


