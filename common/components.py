import base64
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures.thread import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path
import cv2
from open3d import VerbosityLevel, set_verbosity_level
from common import networking
import numpy as np
import open3d
from common import write_bg as wbg, jsonToply
from common.merge import merge
from common.fit_model import fit_return, objective
from common.networking import send_msg, recv_msg
from detector import detect
from common.logger import logger

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def update_sfm_file(args, collect_dir, str_timestamp, DEL_list = []):

    start = time.time()

    tag = "jpg"

    with open(args.parameter, "r") as jsonFile:
        sfm = json.load(jsonFile)

    sfm["root_path"] = "/home/zxj/zxj/distributed-3d-reconstruction/" + os.path.join(collect_dir)

    if args.resolution < 1.0:
        for item in sfm["views"]:
            item["value"]["ptr_wrapper"]["data"]["width"] = int(1920 * args.resolution)
            item["value"]["ptr_wrapper"]["data"]["height"] = int(1080 * args.resolution)

        sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["width"] = int(1920 * args.resolution)
        sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["height"] = int(1080 * args.resolution)
        sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["focal_length"] = int(1920 * args.resolution / 1.118)
        sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["principal_point"] = [int(1920 * args.resolution / 2.06),
                                                                                   int(1080 * args.resolution / 1.93)]
    for item in sfm["views"]:
        item["value"]["ptr_wrapper"]["data"]["filename"] = item["value"]["ptr_wrapper"]["data"]["filename"].replace(
            "png", tag)

    for str1 in DEL_list:
        key = None
        for i in range(len(sfm["views"])):
            if str1 + ".jpg" == sfm["views"][i]["value"]["ptr_wrapper"]["data"]["filename"]:
                key = sfm["views"][i]["key"]
                for j in range(key + 1, len(sfm["views"])):
                    sfm["views"][j]["key"] -= 1
                    sfm["views"][j]["value"]["ptr_wrapper"]["id"] -= 1
                    sfm["views"][j]["value"]["ptr_wrapper"]["data"]["id_view"] -= 1
                    sfm["views"][j]["value"]["ptr_wrapper"]["data"]["id_pose"] -= 1
                    # sfm["views"][j]["value"]["ptr_wrapper"]["data"]["filename"] = str(j-1).zfill(3) + ".jpg"
                sfm["views"].remove(sfm["views"][i])
                break

        if key is not None:
            for i in range(len(sfm["extrinsics"])):
                if key == sfm["extrinsics"][i]["key"]:
                    for j in range(key + 1, len(sfm["extrinsics"])):
                        sfm["extrinsics"][j]["key"] -= 1
                    sfm["extrinsics"].remove(sfm["extrinsics"][i])
                    print(logger(str_timestamp)+"remove view key {}".format(key))
                    break

    sfm["intrinsics"][0]["value"]["polymorphic_id"] = sfm["views"][0]["value"]["ptr_wrapper"]["id"]
    sfm["intrinsics"][0]["value"]["ptr_wrapper"]["id"] = sfm["views"][-1]["value"]["ptr_wrapper"]["id"] + 1

    Path(os.path.join(args.output_dir, str_timestamp + "_output/sfm/matches")).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(args.output_dir, str_timestamp + "_output/sfm/matches/sfm_data.json"), 'w',
              encoding='utf-8') as f:
        json.dump(sfm, f, indent=4)

    print(logger(
        str_timestamp) + f"+ update {bcolors.OKGREEN}SfM configuration{bcolors.ENDC} file in {round((time.time() - start) * 1000)} ms,start SfM pipeline.")

    return int(1920 * args.resolution)


# start to run openMvg + openMvs for foreground
def openMVG(new_cfg, reconstructor, collect_dir, output_dir, str_timestamp):
    start_openMVG = time.time()
    if new_cfg == "1":
        preset = "SEQUENTIAL"
    else:
        preset = "OPENMVG"
    p = subprocess.Popen(
        ["python3", reconstructor, collect_dir, os.path.join(output_dir, str_timestamp + "_output"), "--sfm",
         "sfm_data.bin", "--mvs_dir", "mvs",
         "--preset", preset, "--verbose", str_timestamp])
    p.wait()
    if p.returncode != 0:
        return 0
    return round(time.time() - start_openMVG, 4)


def point_cloud_plit(output_dir, bg_dir, fg_dir, msa):
    start = time.time()
    # convert sfm_data.bin to sfm_data.json
    if sys.platform.startswith('win'):
        cmd = "where"
    else:
        cmd = "which"

    #ret = subprocess.run([cmd, "openMVG_main_SfMInit_ImageListing"], stdout=subprocess.PIPE,
    #                     stderr=subprocess.STDOUT, check=True)
    #OPENMVG_BIN = os.path.split(ret.stdout.decode())[0]

    pChange = subprocess.Popen(
        [os.path.join("openMVG_main_ConvertSfM_DataFormat"), "-i",
         output_dir + "/sfm/sfm_data.bin",
         "-o", output_dir + "/sfm/sfm_data.json"])
    pChange.wait()

    # separate sfm_data.json into background and foreground
    # start_s = time.time()
    L, B, F = wbg.writeBG_FG(output_dir + "/sfm/sfm_data.json", bg_dir, fg_dir, msa, output_dir + "/sfm")
    # print(time.time() - start_s)

    jsonToply.py(output_dir + "/sfm/fg.json", output_dir + "/sfm/fg.ply")
    jsonToply.py(output_dir + "/sfm/bg.json", output_dir + "/sfm/bg.ply")

    # convert bg.json and fg.json back to bin files
    pChange = subprocess.Popen(
        [os.path.join("openMVG_main_ConvertSfM_DataFormat"), "-i", #OPENMVG_BIN
         output_dir + "/sfm/fg.json",
         "-o", output_dir + "/sfm/fg.bin"])
    pChange.wait()

    pChange = subprocess.Popen(
        [os.path.join( "openMVG_main_ConvertSfM_DataFormat"), "-i", # OPENMVG_BIN,
         output_dir + "/sfm/bg.json",
         "-o", output_dir + "/sfm/bg.bin"])
    pChange.wait()
    return round(time.time() - start, 4), L, B, F


def read_log(output_dir):
    log_files = glob.glob(str(Path(output_dir) / '**' / '*.log'), recursive=True)
    times = []
    source_images = []
    if len(log_files) == 1:
        with open(log_files[0]) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        for item in lines:
            if item.__contains__("Depth-map for image"):
                begin = item.find('(') + 1
                end = item.find(')') - 1
                # 2s434m
                time_str = item[begin:end]
                source_images.append(int(item[item.find("using  ") + 7]))
                try:
                    t = int(time_str)
                except:
                    if time_str[1] == 's':
                        if time_str.__contains__('ms'):
                            t = int(time_str[0]) + int(time_str[2:-2]) * 0.001
                        else:
                            t = int(time_str[0]) + int(time_str[2:-1]) * 0.001
                    else:
                        if time_str.__contains__('ms'):
                            t = int(time_str[:-2]) * 0.001
                        else:
                            t = int(time_str[:-1]) * 0.001
                times.append(round(t, 4))
    return times, source_images


def DensifyPointCloud_task(worker, bg, bg_new, cp_time, pred_error, output_dir, n, max_r, str_timestamp, task_setting,
                           server_channel, source_images, view_complexity, static_overhead, a, b, c,  b_ratio, avg_ratio, ba, bb, bc):
    start_densify = time.time()
    items = []

    if bg == 1:
        item_local, item_edge, pre_time = create_task(output_dir, max_r, str_timestamp, task_setting["fg_task_setting"],
                                                      "fg_mvs", [], [], view_complexity, static_overhead, a, b, c,  b_ratio, avg_ratio, ba, bb, bc,
                                                      server_channel=server_channel)
        tag = "[foreground"
        if bg_new == 1:
            item_local, item_edge, pre_time = create_task(output_dir, max_r, str_timestamp,
                                                          task_setting["bg_task_setting"],
                                                          "bg_mvs", item_local, item_edge, view_complexity, static_overhead, a, b, c,  b_ratio, avg_ratio, ba, bb, bc,
                                                          server_channel=server_channel)
            tag += " + background]"
        else:
            tag += "]"
    else:
        item_local, item_edge, pre_time = create_task(output_dir, max_r, str_timestamp, task_setting["task_setting"],
                                                      "mvs", [], [], view_complexity, static_overhead, a, b, c,  b_ratio, avg_ratio, ba, bb, bc,
                                                      server_channel=server_channel)
        tag = ""

    if len(item_local) > 0:
        items.append(item_local)

    if len(item_edge) > 0:
        items.append(item_edge)

    with Pool(len(items)) as p:
        msa = p.map(DensifyPointCloud, items)

    time_densify = {
        "total": round(time.time() - start_densify, 4),
        "pre_total": np.max(pre_time)
    }

    """
    msa = [{   'local': 
                        [
                            {
                                mvs_dir: 
                                mvs_dir + "_pred": 
                                "depth_map_time": 
                                "error": 
                                "static_time":
                                "static_source":
                                "tasks": 
                            }
                        ]
            },
            {
                'edge': {
                
                }
            }
    ]

    """
    print(logger(str_timestamp) + json.dumps(msa, indent=4))

    depth_map_time = {
        "mvs": [],
        "fg_mvs": [],
        "bg_mvs": []
    }

    location = ['local', 'edge']
    for i in range(worker):
        for hist in msa[i][location[i]]:
            for label, value in hist.items():
                pred_error[location[i]][label].append(hist['error'])
                static_overhead[i][label] = round(static_overhead[i][label] * 0.3 + hist['static_time'] * 0.7, 4)
                source_images[label] = round(source_images[label] * 0.2 + hist['source_images'] * 0.8, 4)
                cp_time[location[i]][label].append(round(hist['depth_map_time'], 3))
                depth_map_time[label].append(hist['depth_map_time'])
                break

    #print(logger(
    #    str_timestamp) + f"+ update static latency = {static_overhead[0][label]}, source images = {source_images[label]}")

    print(logger(str_timestamp) + f"+ local {label} pred error = {pred_error['local'][label]}")
    print(logger(str_timestamp) + f"+ edge {label} pred error = {pred_error['edge'][label]}")
    print(logger(str_timestamp) + f"+ average local {label} pred error = {round(np.average(pred_error['local'][label]), 2)}")
    print(logger(str_timestamp) + f"+ average edge {label} pred error = {round(np.average(pred_error['edge'][label]), 2)}")
    print(logger(str_timestamp) + f"+ local depth-map time", cp_time['local'][label])
    print(logger(str_timestamp) + f"+ edge depth-map time", cp_time['edge'][label])
    print(logger(
        str_timestamp) + f"+ {bcolors.OKGREEN}{tag} Densify{bcolors.ENDC}:{bcolors.WARNING}{bcolors.ENDC} in",
          time_densify)

    return time_densify, tag, static_overhead, source_images, depth_map_time


def create_new_dataset_cfg(output_dir, source_dir, n):
    pChange = subprocess.Popen(
        [os.path.join("openMVG_main_ConvertSfM_DataFormat"), "-i",
         output_dir + "/sfm/sfm_data.bin",
         "-o", output_dir + "/sfm/sfm_data.json"])
    pChange.wait()

    with open(output_dir + "/sfm/sfm_data.json", "r") as jsonFile:
        sfm = json.load(jsonFile)

    sfm["structure"] = []

    with open(f"data/parameter/sfm_data_{source_dir[5:]}_{n}.json", 'w', encoding='utf-8') as f:
        json.dump(sfm, f, indent=4)


def profile_bg(str_timestamp, reconstructor, bg_new, images_IDS, collect_dir, output_dir, fg_dir, bg_dir, subNet,
               obs_profile, profile=True):
    start_profile = time.time()
    settings = [
        {
            "conf_thres": 0.25,
            "class": ['person']
        },
        {
            "conf_thres": 0.25,
            "class": ['person', 'tv', 'couch', 'table']
        },
        {
            "conf_thres": 0.25,
            "class": None
        },
        {
            "conf_thres": 0.005,
            "class": None
        }
    ]
    profile_resolution = [192, 240, 336, 432, 576, 864]
    profile_resolution_base = [192, 240, 288, 336, 384, 432, 480, 576, 672, 768, 864, 960] # [192, 240, 288, 336, 384, 432, 480, 576, 672, 768, 864, 960]
    ratio = []
    base = True
    ba, bb, bc, b_source = 0, 0, 0, 0
    if profile is True:
        for setting in settings:
            items = []
            msa = []
            ratios = []
            object_number = []
            for ID in images_IDS:
                items.append((0, os.path.join(collect_dir, ID + ".jpg"), os.path.join(fg_dir, ID + ".jpg"),
                              os.path.join(bg_dir, ID + ".jpg"), subNet.model, subNet.imgsz, subNet.stride, subNet.half,
                              subNet.device, setting['conf_thres'], setting['class'], subNet.opt))

            with ThreadPoolExecutor(max_workers=len(items)) as executor:
                results = executor.map(detect, items)

            for result in results:
                msa.append(result[0])
                ratios.append(result[1])
                object_number.append(result[2])

            ratio.append(round(np.average(ratios), 2))

            obs_profile[str(ratio[-1])] = {}

            print(logger(str_timestamp) + "+ find target views = {}".format(len(msa)))
            print(logger(str_timestamp) + "+ number of RoI boxes = {}, avg={}".format(object_number,
                                                                                      round(np.average(object_number),
                                                                                            2)))
            print(logger(
                str_timestamp) + f"+ ratio of RoI areas = {ratio}, avg={bcolors.OKGREEN}{ratio[-1]}{bcolors.ENDC}")

            time_split, L, B, F = point_cloud_plit(output_dir, bg_dir, fg_dir, msa)
            time_split += opeMVG2openMVS(reconstructor, fg_dir, output_dir, "fg.bin", "fg_mvs")

            resolution = profile_resolution if not base else profile_resolution_base
            print(logger(str_timestamp) + f"+ start to profile ratio = {ratio[-1]}, resolution = {resolution}")
            b_source_image = []
            for r in resolution:
                p = subprocess.Popen(
                    ["python3", reconstructor, collect_dir, output_dir, "--sfm", "sfm_data.bin", "--mvs_dir", "fg_mvs",
                     "--preset", "DensifyPointCloud", "--tasks", "[000]-[006]", "--resolution", str(r), "--do_fuse",
                     "0"])
                p.wait()
                times, source_images = read_log(output_dir + "/fg_mvs")
                obs_profile[str(ratio[-1])][str(r)] = np.sum(times)
                b_source_image.append(np.average(source_images))
                test = os.listdir(output_dir + "/fg_mvs")
                for item in test:
                    if item.endswith(".dmap") or item.endswith(".log"):
                        os.remove(os.path.join(output_dir + "/fg_mvs", item))
            if base:
                ba, bb, bc = fit_return([obs_profile[str(ratio[-1])][str(r)] for r in resolution], resolution)
                b_source = np.average(b_source_image)
                print(logger(
                    str_timestamp) + f"+ fitted parameters ratio {ratio[-1]}, a={ba},b={bb},c={bc}")
                base = False

            print(logger(str_timestamp), json.dumps(obs_profile, indent=4))

        print(logger(
            str_timestamp) + f"+ profile in {round(time.time() - start_profile, 4)}")

    return ratio, obs_profile, ba, bb, bc, b_source


def fit_profile(str_timestamp, obs_profile, a, b, c):
    profile_resolution_base = [192, 240, 288, 336, 384, 432, 480, 576, 672, 768, 864, 960]
    diff_ratio = []
    profile_resolution = []
    ratio = [item for item, value in obs_profile.items()]
    for k in range(len(profile_resolution_base)):
        diff_time = []
        for i in range(len(ratio)):
            for j in range(i, len(ratio)):
                if i != j and str(profile_resolution_base[k]) in obs_profile[ratio[i]] and str(
                        profile_resolution_base[k]) in \
                        obs_profile[
                            ratio[j]]:
                    p = np.abs(round(obs_profile[ratio[i]][str(profile_resolution_base[k])] - obs_profile[ratio[j]][
                        str(profile_resolution_base[k])], 3))
                    g = np.abs(round(float(ratio[i]) - float(ratio[j]), 3) * 100)
                    diff_time.append(p / g)
        if len(diff_time) > 0:
            diff_ratio.append(np.average(diff_time))
            profile_resolution.append(profile_resolution_base[k])

    if len(profile_resolution) > 0:
        a1, b1, c1 = fit_return(diff_ratio, profile_resolution)
        if a == 0:
            a[0] = a1
            b[0] = b1
            c[0] = c1
        else:
            a[0] = 0.8 * a[0] + a1 * 0.2
            b[0] = 0.8 * b[0] + b1 * 0.2
            c[0] = 0.8 * c[0] + c1 * 0.2
        print(logger(str_timestamp) + f"+ {bcolors.OKCYAN}fitted parameters{bcolors.ENDC} a={a},b={b},c={c}")

    return a, b, c


def profile(str_timestamp, reconstructor, collect_dir, output_dir, obs_profile, profile=True):

    profile_resolution = []
    for item, value in obs_profile.items():
        profile_resolution.append(int(item))

    start_profile = time.time()
    if profile:
        print(logger(str_timestamp) + f"+ start to profile {profile_resolution}")
        time_avg_small_obs = []
        for r in profile_resolution:
            p = subprocess.Popen(
                ["python3", reconstructor, collect_dir, output_dir, "--sfm", "sfm_data.bin", "--mvs_dir", "mvs",
                 "--preset", "DensifyPointCloud", "--tasks", "[000]-[006]", "--resolution", str(r), "--do_fuse", "0"])
            p.wait()
            time_avg_small_obs.append(np.average(read_log(output_dir + "/mvs")[0]))
            test = os.listdir(output_dir + "/mvs")
            for item in test:
                if item.endswith(".dmap") or item.endswith(".log"):
                    os.remove(os.path.join(output_dir + "/mvs", item))
    else:
        time_avg_small_obs = []
        for item, value in obs_profile.items():
            time_avg_small_obs.append(value)

    a, b, c = fit_return(time_avg_small_obs, profile_resolution)
    print(logger(str_timestamp) + f"+ fitted parameters a={a},b={b},c={c}, in {round(time.time() - start_profile, 4)}")
    print(time_avg_small_obs)

    for i in range(len(profile_resolution)):
        if obs_profile[str(profile_resolution[i])] == 0:
            obs_profile[str(profile_resolution[i])] = time_avg_small_obs[i]
        else:
            obs_profile[str(profile_resolution[i])] = obs_profile[str(profile_resolution[i])] * 0.3 + \
                                                      time_avg_small_obs[i] * 0.7

    return a, b, c, obs_profile


def opeMVG2openMVS(reconstructor, input_dir, output_dir, sfm_file, mvs_dir):
    start = time.time()
    p = subprocess.Popen(
        ["python3", reconstructor, input_dir, output_dir, "--sfm", sfm_file, "--mvs_dir", mvs_dir,
         "--preset", "MVG_MVS"])
    p.wait()
    if p.returncode != 0:
        return 0
    return round(time.time() - start, 4)


def create_task(output_dir, max_r, str_timestamp, task_setting, mvs_dir, item_local, item_edge, view_complexity, static_overhead, a, b, c,  b_ratio, avg_ratio, ba, bb, bc, server_channel=None):
    dir_mvs = output_dir + "/" + mvs_dir
    dir_img = output_dir + "/" + mvs_dir + "/images"

    pred_times = []
    for i in range(len(task_setting)):
        if task_setting[i]["task"] == "":
            continue
        channel = None if task_setting[i]["type"] == "local" else server_channel[
            task_setting[i]["server"]]

        task_begin = int(task_setting[i]["task"][1:4])
        task_end = int(task_setting[i]["task"][7:10])

        pred_time = 0
        if mvs_dir == "mvs":
            for t in range(task_begin, task_end + 1):
                pred_time += objective(int(max_r), a[i], b[i], c[i]) * view_complexity[i][t]
        else:
            r_diff = round(avg_ratio - b_ratio, 3) * 100

            diff = round(objective(int(max_r), a[i], b[i], c[i]), 4)

            from_base = round(objective(int(max_r), ba, bb, bc), 4)

            scale = 1 # source_images["fg_mvs"] / b_source

            pred_time = round((diff * r_diff + from_base) * scale, 4)

       # task_number = int(task_setting[i]["task"][7:10]) - int(task_setting[i]["task"][1:4]) + 1

        pred_avg_depth_map_time = round(pred_time/(task_end - task_begin + 1), 4)

        pred_time = round(pred_time + static_overhead[i][mvs_dir], 4)

        pred_times.append(pred_time)

        if task_setting[i]["type"] == "local":
            item_local.append((
                max_r, task_setting[i]["task"], dir_mvs, dir_img,
                task_setting[i]["type"], str_timestamp, channel, mvs_dir, pred_time, pred_avg_depth_map_time))
        elif task_setting[i]["type"] == "edge":
            item_edge.append((
                max_r, task_setting[i]["task"], dir_mvs, dir_img,
                task_setting[i]["type"], str_timestamp, channel, mvs_dir, pred_time, pred_avg_depth_map_time))

    return item_local, item_edge, pred_times


def do_merge(fg_path, bg_path, output_path):
    start = time.time()
    merge(fg_path,  bg_path, output_path)
    return round(time.time() - start, 4)


def fuse(max_r, reconstructor, collect_dir, output_dir, mvs_dir, str_timestamp):

    start_fuse = time.time()

    p = subprocess.Popen(
        ["DensifyPointCloud", "scene.mvs", "--dense-config-file", "Densify.ini", "--resolution-level", "0",
         "--max-resolution",
         max_r, "--min-resolution", "320",
         "-w", output_dir + "/" + mvs_dir, "--tasks", "1", "--task-number", "1", "--do-fuse", "1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    p.wait()

    return round(time.time() - start_fuse, 4)


def eva(str_timestamp, output_path, truth_path):

    set_verbosity_level(VerbosityLevel.Error)

    gt = open3d.read_point_cloud(truth_path)
    pr = open3d.read_point_cloud(output_path)

    ths = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]

    precisions = []
    recalls = []
    fscores = []

    for th in ths:
        d1 = open3d.compute_point_cloud_to_point_cloud_distance(gt, pr)
        d2 = open3d.compute_point_cloud_to_point_cloud_distance(pr, gt)

        if len(d1) and len(d2):
            recall = float(sum(d < th for d in d2)) / float(len(d2))
            precision = float(sum(d < th for d in d1)) / float(len(d1))

        if recall + precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
            precision = 0
            recall = 0

        precisions.append(round(precision, 3))
        recalls.append(round(recall, 3))
        fscores.append(round(fscore, 3))

    print(logger(str_timestamp) + "+ precisions=", precisions)
    print(logger(str_timestamp) + "+ recalls=", recalls)
    print(logger(str_timestamp) + "+ fscores=", fscores)
    return precisions, recalls, fscores


def save_time2(time_hist, str_timestamp, precisions, recalls, fscores):

    with open("../time.json", "r") as jsonFile:
        time_file = json.load(jsonFile)

    for i in range(len(time_file["timeList"])):
        if time_file["timeList"][i]["id"] == str_timestamp:
            time_file["timeList"][i]["eva"] = {
                "precision": precisions,
                "recall": recalls,
                "f1": fscores
            }
            time_file["time"] = time_hist

            break

    with open('../time.json', 'w', encoding='utf-8') as f:
        json.dump(time_file, f, indent=4)

    print(logger(str_timestamp) + "+ [total time={}, network={}, openMVG={}, Densify={}]".format(
        bcolors.OKGREEN + str(time_hist["total_time"]) + bcolors.ENDC
          , round(time_hist["network_delay"], 4), round(time_hist["time_openMVG"], 4),
          time_hist["time_densify"]["total"]))


def save_time(str_timestamp, compress, time_total, network_delay, time_openMVG, time_densify, time_fuse, precisions,
              recalls, fscores, time_merge):

    with open("time.json", "r") as jsonFile:
        time_file = json.load(jsonFile)

    for i in range(len(time_file["timeList"])):
        if time_file["timeList"][i]["id"] == str_timestamp:
            time_file["timeList"][i]["compression"] = compress
            time_file["timeList"][i]["total time"] = round(time_total, 4)
            time_file["timeList"][i]["network delay"] = round(network_delay, 4)
            time_file["timeList"][i]["openMVG time"] = round(time_openMVG, 4)
            time_file["timeList"][i]["Densify point cloud time"] = time_densify
            time_file["timeList"][i]["Fuse time"] = round(time_fuse, 4)
            if time_merge != 0:
                time_file["timeList"][i]["Merge time"] = round(time_merge, 4)
            time_file["timeList"][i]["eva"] = {
                "precision": precisions,
                "recall": recalls,
                "f1": fscores
            }

            break

    with open('../time.json', 'w', encoding='utf-8') as f:
        json.dump(time_file, f, indent=4)

    print(logger(str_timestamp) + "[total time={}, network={}, openMVG={}, Densify={}]".format(bcolors.OKGREEN+str(round(time_total, 4))+bcolors.ENDC, round(network_delay, 4), round(time_openMVG, 4), time_densify["total"]))

    #with open("time_all.json", "r") as jsonFile:
    #    time_all_file = json.load(jsonFile)

   # time_all_file["timeList"].append(time_file["timeList"][i])

   # with open('time_all.json', 'w', encoding='utf-8') as f:
   #     json.dump(time_all_file, f, indent=4)


def DensifyPointCloud(items):

    densify_start = time.time()
    time_densify = {}

    for values in items:
        max_r, tasks, dir_mvs, dir_img, t_type, timestamp, server_channel, mvs_dir, pred_time, pred_avg_depth_map_time = values

        if t_type == "local":
            p = subprocess.Popen(
                ["DensifyPointCloud", "scene.mvs", "--dense-config-file", "Densify.ini", "--resolution-level", "0", "--max-resolution",
                 max_r, "--min-resolution", "320",
                 "-w", dir_mvs, "--tasks", tasks, "--do-fuse", "0"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

            p.wait()

            depth_map_time, source_images = read_log(dir_mvs)

            if t_type not in time_densify:
                time_densify[t_type] = []

            actual_time = round(time.time() - densify_start, 4)
            densify_start = time.time()

            error = round(np.abs(pred_avg_depth_map_time -np.average(depth_map_time)), 4)
            time_densify[t_type].append({
                mvs_dir: actual_time,
                mvs_dir + "_pred": str(pred_time),
                "depth_map_time": round(np.average(depth_map_time), 4),
                "times": depth_map_time,
                "pred_avg_depth_map_time": round(pred_avg_depth_map_time, 4),
                "error": error,
                "static_time": round(np.abs(np.sum(depth_map_time) - actual_time), 4),
                "source_images": round(np.average(source_images), 4),
                "tasks": tasks
            })
        else:
            try:
                # send image
                f = []  # image files
                p = Path(dir_img)  # os-agnostic
                f += glob.glob(str(p / '**' / '*.jpg'), recursive=True)

                data_string = []
                files = []
                for item in f:
                    with open(item, 'rb') as file:
                        data = file.read()
                    data_string.append(base64.encodebytes(data).decode("utf-8"))
                    files.append(item[-7:])

                with open(dir_mvs + "/scene.mvs", 'rb') as file:
                    mvs = file.read()

                # with open(dir_mvs + "/Densify.ini", 'rb') as file:
                #    ini = file.read()

                msg = {"data": data_string, "files": files, "type": "images", "mvs_dir": mvs_dir,
                       "str_timestamp": timestamp, "max_r": max_r,
                       "mvs": base64.encodebytes(mvs).decode("utf-8")}
                send_msg(server_channel, json.dumps(msg).encode("utf-8"))

                # send task
                # [009]-[015]
                task_number = int(tasks[7:10]) - int(tasks[1:4]) + 1

                msg = {"tasks": tasks,  "type": "task_info", "mvs_dir": mvs_dir,
                       "str_timestamp": timestamp, "max_r": max_r}
                send_msg(server_channel, json.dumps(msg).encode("utf-8"))

                # recieve dethpmap
                task = 0
                while True:
                    data = recv_msg(server_channel)
                    info = json.loads(str(data.decode('utf-8')))

                    with open(dir_mvs + "/" + info["file"], 'wb') as file:
                        file.write(base64.b64decode(info["data"]))
                    task += 1
                    if task >= task_number:
                        break

                data = recv_msg(server_channel)
                info = json.loads(str(data.decode('utf-8')))

                if t_type not in time_densify:
                    time_densify[t_type] = []

                actual_time = info["actual_time"]
                error = round(np.abs(pred_avg_depth_map_time -np.average(info["depth_map_time"])), 4)

                time_densify[t_type].append({
                    mvs_dir: actual_time,
                    mvs_dir + "_pred": str(pred_time),
                    "depth_map_time": round(np.average(info["depth_map_time"]), 4),
                    "pred_avg_depth_map_time": round(pred_avg_depth_map_time, 4),
                    "times": info["depth_map_time"],
                    "error": error,
                    "static_time": round(np.abs(np.sum(info["depth_map_time"]) - actual_time), 4),
                    "source_images": round(np.average(info["source_images"]), 4),
                    "tasks": tasks
                })
                densify_start = time.time()
                break

            except:
                print(traceback.format_exc())
                print(logger() + "port: disconnected")
                return time_densify

    return time_densify


def encoding(items):
    img_file_name, data_dir, image_dir, collect_dir, resolution, compress = items
    shutil.copy2(os.path.join(data_dir, image_dir, img_file_name),
                 os.path.join(collect_dir, image_dir + ".png"))

    src = cv2.imread(os.path.join(collect_dir, image_dir + ".png"))
    cv2.imwrite(os.path.join(collect_dir, image_dir + ".jpg"), src, [int(cv2.IMWRITE_JPEG_QUALITY), compress])
    os.remove(os.path.join(collect_dir, image_dir + ".png"))


def local_dataloader(str_timestamp, resolution, compress, data_dir, collect_dir, dirs):
    start = time.time()
    s = str_timestamp + str(dirs)
    items = []
    for image_dir in dirs:
        items.append((str_timestamp + ".png", data_dir, image_dir, collect_dir, resolution, compress))

    with Pool(len(items)) as p:
        p.map(encoding, items)

    return round(time.time() - start, 4), s, dirs


def remote_dataloader(user_channel, collect_dir):
    start = time.time()
    msg = {"status": "send"}
    send_msg(user_channel, json.dumps(msg).encode("utf-8"))

    # getting image
    try:
        data = recv_msg(user_channel)
        info = json.loads(str(data.decode('utf-8')))

        s = "get images = ["
        for i in range(len(info["files"])):
            with open(os.path.join(collect_dir, info["files"][i] + ".png"), 'wb') as file:
                file.write(base64.b64decode(info["data"][i]))
            s += info["files"][i] + " "
        s += "]"
        print(s)

        network_delay = round(time.time() - start, 4)

        msg = {"status": "ok"}
        send_msg(user_channel, json.dumps(msg).encode("utf-8"))
    except:
        print(traceback.format_exc())
    return network_delay

