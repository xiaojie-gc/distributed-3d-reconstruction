import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from common.components import bcolors, point_cloud_plit, opeMVG2openMVS, DensifyPointCloud_task, fuse, profile_bg, \
    fit_profile
from common.fit_model import fit_return
from common.logger import logger
from detector import detect


def bg(args, str_timestamp, a, b, c, bg_new, reconstructor, images_dirs, collect_dir, fg_dir, bg_dir, output_dir,
       subNet, b_ratio, ba, bb, bc, source_images, b_source, static_overhead, cp_time, pred_error, task_setting,
       server_channel, view_complexity, obs_profile):
    items = []
    msa = []
    ratio = []
    object_number = []
    start_sub = time.time()

    val = random.uniform(0, 1)

    # if val <= 0.35:
    all_classes = ['person', 'tv', 'couch', 'table']
    # elif 0.35 < val <= 0.65:
    #    all_classes = ['person']
    # else:
    #    all_classes = None

    for ID in images_dirs:
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
    print(logger(str_timestamp) + f"+ RoI detection (ratio={avg_ratio}) in", round(time.time() - start_sub, 4))

    ######################################################################################
    time_split, L, B, F = point_cloud_plit(output_dir, bg_dir, fg_dir, msa)
    print(time_split)
    time_split += opeMVG2openMVS(reconstructor, fg_dir, output_dir, "fg.bin", "fg_mvs")
    if bg_new == 1:
        time_split += opeMVG2openMVS(reconstructor, bg_dir, output_dir, "bg.bin", "bg_mvs")
    print(logger(str_timestamp) + f"+ {bcolors.OKGREEN}openMVG-to-openMVS{bcolors.ENDC} in", round(time_split, 4),
          "{}/{}/{}".format(L, B, F))
    #####################################################################################

    #print(logger(str_timestamp),
    #      f"+ scale based on number of source images = {bcolors.OKBLUE}{round(source_images['fg_mvs'] / b_source, 4)}{bcolors.ENDC}")

    #print(logger(str_timestamp), f"({diff} * {r_diff} + {from_base})* {scale}  +{static_overhead['fg_mvs']}")

    time_densify, tag, static_overhead, source_images, depth_map_time = DensifyPointCloud_task(
        args.worker,
        args.bg, args.bg_new, cp_time,
        pred_error, output_dir, args.n,
        args.max_r, str_timestamp,
        task_setting, server_channel,
        source_images,
        view_complexity, static_overhead, a, b, c,  b_ratio, avg_ratio, ba, bb, bc)

    """
    if str(avg_ratio) not in obs_profile:
        obs_profile[str(avg_ratio)] = {
            args.max_r: depth_map_time * (1 / scale)
        }
    else:
        if args.max_r in obs_profile[str(avg_ratio)]:
            obs_profile[str(avg_ratio)][args.max_r] = obs_profile[str(avg_ratio)][
                                                          args.max_r] * 0.3 + depth_map_time * 0.7 * (1 / scale)
        else:
            obs_profile[str(avg_ratio)][args.max_r] = depth_map_time * (1 / scale)
    """
    if avg_ratio == b_ratio:
        all_resolution = [int(item) for item, value in obs_profile[str(b_ratio)].items()]
        ba, bb, bc = fit_return([obs_profile[str(b_ratio)][str(r)] for r in all_resolution], all_resolution)
        print(logger(
            str_timestamp) + f"+ {bcolors.OKCYAN}fitted parameters{bcolors.ENDC} ratio {b_ratio}, ba={ba},bb={bb},bc={bc}")

    print(logger(str_timestamp), obs_profile)

    a, b, c = fit_profile(str_timestamp, obs_profile, a, b, c)

    return time_densify, a, b, c, obs_profile