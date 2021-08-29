import random
from _thread import start_new_thread
from threading import Lock
import argparse
from common.functions import *
import psutil
import numpy as np

from common.networking import user_discovery, server_discovery
from detector import Server, detect
from common.DTU import prepare_img_dir
from common.fit_model import fit_return, objective
from initial import *

parser = argparse.ArgumentParser(description='Example command:')
args = initial_parameters(parser)
task_setting = initial_task_setting(args, task_setting_file="task.json")
logger_parameters(args)

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
subNet = Server(args)
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
    "fg_mvs": [],
    "bg_mvs": [],
    "mvs": []
}

cp_time = {
    "fg_mvs": [],
    "bg_mvs": [],
    "mvs": []
}

a, b, c, ba, bb, bc = 0, 0, 0, 0, 0, 0
b_ratio, b_source = 0, 5

if args.bg == 0:
    obs_profile = {
        "192": 0,
        "240": 0,
        "336": 0,
        "432": 0
    }
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

source_images = {
    "fg_mvs": 5,
    "bg_mvs": 5,
    "mvs": 5
}

static_overhead = {
    "mvs": 0.35,
    "fg_mvs": 0.35,
    "bg_mvs": 0.35
}

while True:
    args.max_r = all_r[inx] if True else random.randint(0, 8)
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
            a, b, c, obs_profile = profile(str_timestamp, args.reconstructor, collect_dir, output_dir, obs_profile)
        else:
            a, b, c, obs_profile, ba, bb, bc, b_ratio, b_source = profile_bg(str_timestamp, args.reconstructor,
                                                                             args.bg_new, dirs, collect_dir, output_dir,
                                                                             fg_dir, bg_dir,
                                                                             subNet, obs_profile, profile=True)

        args.profile = 0
    ####################################################################################################
    time_merge = 0
    if args.bg == 1:
        images_IDS = dirs
        items = []
        msa = []
        ratio = []
        object_number = []
        start_sub = time.time()

        val = random.uniform(0, 1)

        #if val <= 0.35:
        all_classes = ['person', 'tv', 'couch', 'table']
        #elif 0.35 < val <= 0.65:
        #    all_classes = ['person']
        #else:
        #    all_classes = None

        for ID in dirs:
            items.append((0, os.path.join(collect_dir, ID + ".jpg"), os.path.join(fg_dir, ID + ".jpg"),
                          os.path.join(bg_dir, ID + ".jpg"), subNet.model, subNet.imgsz, subNet.stride, subNet.half,
                          subNet.device, random.uniform(0.25, 0.25), all_classes, subNet.opt))

        with ThreadPoolExecutor(max_workers=len(items)) as executor:
            results = executor.map(detect, items)

        for result in results:
            msa.append(result[0])
            ratio.append(result[1])
            object_number.append(result[2])

        avg_ratio = round(np.average(ratio), 2)

        m.acquire()
        #print(logger(str_timestamp) + "+ find target views = {}".format(len(msa)))

        #print(logger(str_timestamp) + "+ number of RoI boxes = {}, avg={}".format(object_number,
        #                                                                          round(np.average(object_number), 2)))
       # print(logger(
        #    str_timestamp) + f"+ ratio of RoI areas = {ratio}, avg={bcolors.OKGREEN}{avg_ratio}{bcolors.ENDC}")

        print(logger(str_timestamp) + f"+ RoI detection (ratio={avg_ratio}) in", round(time.time() - start_sub, 4),
              f"{bcolors.HEADER}[CPU:{round(np.average(cpus), 1)}%]{bcolors.ENDC}")
        cpus = []
        m.release()
        ######################################################################################
        time_split, L, B, F = point_cloud_plit(output_dir, bg_dir, fg_dir, msa)
        print(time_split)
        time_split += opeMVG2openMVS(args.reconstructor, fg_dir, output_dir, "fg.bin", "fg_mvs")
        if args.bg_new == 1:
            time_split += opeMVG2openMVS(args.reconstructor, bg_dir, output_dir, "bg.bin", "bg_mvs")
        m.acquire()
        print(logger(str_timestamp) + f"+ {bcolors.OKGREEN}openMVG-to-openMVS{bcolors.ENDC} in", round(time_split, 4), "{}/{}/{}".format(L,B,F),
              f"{bcolors.HEADER}[CPU:{round(np.average(cpus),1)}%]{bcolors.ENDC}")
        cpus = []
        m.release()
        ######################################################################################
        # estimate computation latency
        r_diff = round(avg_ratio - b_ratio, 3) * 100

        diff = round(objective(int(args.max_r), a, b, c), 4)

        from_base = round(objective(int(args.max_r), ba, bb, bc), 4)

        scale = source_images["fg_mvs"] / b_source

        p = round((diff * r_diff + from_base) * scale + static_overhead["fg_mvs"], 4)

        print(logger(str_timestamp), f"+ scale based on number of source images = {bcolors.OKBLUE}{round(source_images['fg_mvs']/b_source, 4)}{bcolors.ENDC}")

        print(logger(str_timestamp), f"({diff} * {r_diff} + {from_base})* {scale}  +{static_overhead['fg_mvs']}")

        time_densify, msa, cp_time, tag, static_overhead, source_images, depth_map_time = DensifyPointCloud_task(args.bg, args.bg_new, cp_time,
                                                                                  pred_error, output_dir,
                                                                                  args.n, args.max_r, str_timestamp,
                                                                                  task_setting,
                                                                                  server_channel,
                                                                                  source_images,
                                                                                  static_overhead,
                                                                                  pred_fg_avg_time=p,
                                                                                  pred_bg_avg_time=None,
                                                                                  pred_avg_time=None)

        if str(avg_ratio) not in obs_profile:
            obs_profile[str(avg_ratio)] = {
                args.max_r: depth_map_time * (1/scale)
            }
        else:
            if args.max_r in obs_profile[str(avg_ratio)]:
                obs_profile[str(avg_ratio)][args.max_r] = obs_profile[str(avg_ratio)][args.max_r] * 0.3 + depth_map_time * 0.7 * (1/scale)
            else:
                obs_profile[str(avg_ratio)][args.max_r] = depth_map_time * (1/scale)

        if avg_ratio == b_ratio:
            all_resolution = [int(item) for item, value in obs_profile[str(b_ratio)].items()]
            ba, bb, bc = fit_return([obs_profile[str(b_ratio)][str(r)] for r in all_resolution], all_resolution)
            print(logger(
                str_timestamp) + f"+ {bcolors.OKCYAN}fitted parameters{bcolors.ENDC} ratio {b_ratio}, ba={ba},bb={bb},bc={bc}")

        print(logger(str_timestamp), obs_profile)

        diff_ratio = []
        all_resolution = [192, 240, 288, 336, 384, 432, 480, 576, 672, 768, 864, 960]
        profile_resolution = []
        ratio = [item for item, value in obs_profile.items()]
        #print(logger(str_timestamp), ratio)
        for k in range(len(all_resolution)):
            diff_time = []
            for i in range(len(ratio)):
                for j in range(i, len(ratio)):
                    if i != j and str(all_resolution[k]) in obs_profile[ratio[i]] and str(all_resolution[k]) in obs_profile[ratio[j]]:
                        p = np.abs(round(obs_profile[ratio[i]][str(all_resolution[k])] - obs_profile[ratio[j]][
                            str(all_resolution[k])], 3))
                        g = np.abs(round(float(ratio[i]) - float(ratio[j]), 3) * 100)
                        diff_time.append(p / g)
            if len(diff_time) > 0:
                diff_ratio.append(np.average(diff_time))
                profile_resolution.append(all_resolution[k])

        # print(logger(str_timestamp), f"+ add {profile_resolution}")
        if len(profile_resolution) > 0:
            a1, b1, c1 = fit_return(diff_ratio, profile_resolution)
            a = 0.8 * a + a1 * 0.2
            b = 0.8 * b + b1 * 0.2
            c = 0.8 * c + c1 * 0.2
            print(logger(str_timestamp) + f"+ {bcolors.OKCYAN}fitted parameters{bcolors.ENDC} a={a},b={b},c={c}")

        ######################################################################################
        time_fuse = fuse(args.max_r, args.reconstructor, collect_dir, output_dir, "fg_mvs", str_timestamp)
        if args.bg_new == 1:
            time_fuse += fuse(args.max_r, args.reconstructor, collect_dir, output_dir, "bg_mvs", str_timestamp)
        m.acquire()
        print(logger(str_timestamp) + f"+ {bcolors.OKGREEN}fuse{bcolors.ENDC} in", round(time_fuse, 4),
              f"{bcolors.HEADER}[CPU:{round(np.average(cpus),1)}%]{bcolors.ENDC}")
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
                              output_path="data/" + result_dir + "/" + str_timestamp +"_scene_dense.ply")
        m.acquire()
        print(logger(str_timestamp) + f"+ {bcolors.OKBLUE}merge{bcolors.ENDC} in", time_merge)
        cpus = []
        m.release()
    else:
        ######################################################################################
        opeMVG2openMVS(args.reconstructor, collect_dir, output_dir, "sfm_data.bin", "mvs")
        ######################################################################################
        # estimate computation latency
        pred_avg_time = round(objective(int(args.max_r), a, b, c) * int(args.n) + static_overhead["mvs"], 4)

        time_densify, msa, cp_time, tag, static_overhead, source_images, depth_map_time = DensifyPointCloud_task(args.bg, args.bg_new, cp_time,
                                                                                  pred_error, output_dir, args.n,
                                                                                  args.max_r, str_timestamp,
                                                                                  task_setting, server_channel,
                                                                                  source_images,
                                                                                  static_overhead,
                                                                                  pred_fg_avg_time=None,
                                                                                  pred_bg_avg_time=None,
                                                                                  pred_avg_time=pred_avg_time)

        if args.max_r not in obs_profile:
            obs_profile[args.max_r] = depth_map_time/7
        else:
            obs_profile[args.max_r] = obs_profile[args.max_r] * 0.3 + 0.7 * depth_map_time /7

        a, b, c, obs_profile = profile(str_timestamp, args.reconstructor, collect_dir, output_dir, obs_profile, profile=False)

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
    #precisions, recalls, fscores = eva(str_timestamp, output_path="data/"+result_dir+"/" + str_timestamp + "_scene_dense.ply", truth_path="data/no_background_sub_960_2/00000_scene_dense.ply")

    save_time(str_timestamp, args.compress, time_total, network_delay, time_openMVG, time_densify, time_fuse,
              None,
              None, None, time_merge)

    # timestamp += 1

    inx += 1


