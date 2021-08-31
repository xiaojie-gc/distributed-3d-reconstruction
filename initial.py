import json
import os
import shutil
from pathlib import Path

from common.components import logger, bcolors


def logger_parameters(args):
    print(logger() + '#' * 100)
    print(logger() + 'Multi-view 3D Reconstruction')
    print(logger() + '@ Xiaojie Zhang')
    print(logger() + "To start: please modify the parameters settings in file 'start.sh'")
    print(logger() + "One server + No new background example: ./start.sh 1")
    print(logger() + "Two servers + new background example: ./start.sh 2")
    print(logger() + "Two servers + No background subtraction example: ./start.sh 3")
    print(logger(
        ) + f'{bcolors.OKCYAN}--bg{bcolors.ENDC} = {bcolors.WARNING}{args.bg}{bcolors.ENDC}: 1 if do background subtraction else 0.')
    print(logger(
        ) + f'{bcolors.OKCYAN}--bg_new{bcolors.ENDC} = {bcolors.WARNING}{args.bg_new}{bcolors.ENDC}: 1 if use new background else 0.')
    print(logger(
        ) + f'{bcolors.OKCYAN}--worker{bcolors.ENDC} = {bcolors.WARNING}{args.worker}{bcolors.ENDC}: number of servers, default=2')
    print(logger(
        ) + f'{bcolors.OKCYAN}--max_r{bcolors.ENDC} = {bcolors.WARNING}{args.max_r + "(" + str(round(int(args.max_r) / 1920, 1)) + ")"}{bcolors.ENDC}: Densify resolution, please notice that openMVG uses the default resolution (1920,1080)')
    print(logger(
        ) + f'{bcolors.OKCYAN}--n{bcolors.ENDC} = {bcolors.WARNING}{args.n}{bcolors.ENDC}: number of views for reconstruction, default=7')
    print(logger(
        ) + f"{bcolors.OKCYAN}--remote{bcolors.ENDC} = {bcolors.WARNING}{args.remote}{bcolors.ENDC}: 'yes' if get images from a remote user else 'no', default='no;")
    print(logger(
        ) + f"{bcolors.OKCYAN}--edge_fg{bcolors.ENDC} = {bcolors.WARNING}{args.edge_fg}{bcolors.ENDC} :  FG Densify task allocation for edge")
    print(logger(
        ) + f"{bcolors.OKCYAN}--edge_bg{bcolors.ENDC} = {bcolors.WARNING}{args.edge_bg}{bcolors.ENDC} :  BG Densify task allocation for edge")
    print(logger(
        ) + f"{bcolors.OKCYAN}--edge{bcolors.ENDC} = {bcolors.WARNING}{args.edge if args.bg == 0 else 'none'}{bcolors.ENDC} :  none-background subtraction, Densify task allocation for edge")
    print(logger(
        ) + f"{bcolors.OKCYAN}--local_fg{bcolors.ENDC} = {bcolors.WARNING}{args.local_fg}{bcolors.ENDC} :  FG Densify task allocation for local")
    print(logger(
        ) + f"{bcolors.OKCYAN}--local_bg{bcolors.ENDC} = {bcolors.WARNING}{args.local_bg}{bcolors.ENDC} :  BG Densify task allocation for local")
    print(logger(
        ) + f"{bcolors.OKCYAN}--local{bcolors.ENDC} = {bcolors.WARNING}{args.local if args.bg == 0 else 'none'}{bcolors.ENDC} :  none-background subtraction, Densify task allocation for local")
    print(logger() + '#' * 100)


def initial_parameters(parser):
    ######################
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5x.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    ###################################

    parser.add_argument('--time', type=int, default=0,
                        help="the directory which contains the pictures set.")
    parser.add_argument('--source_dir', type=str, default='data/originals',
                        help="the directory which contains the pictures set.")
    parser.add_argument('--data_dir', type=str, default='data/originals',
                        help="the directory which contains the pictures set.")
    parser.add_argument('--data_collect_dir', type=str, default='data/collect',
                        help="the directory which contains the pictures set.")
    parser.add_argument('--fg_dir', type=str, default='data/fg',
                        help="the directory which contains the foreground pictures set.")
    parser.add_argument('--bg_dir', type=str, default='data/bg',
                        help="the directory which contains the background pictures set.")
    parser.add_argument('--output_dir', type=str, default='data/gold_results',
                        help="the directory which contains the final results.")
    parser.add_argument('--parameter', type=str, default='data/parameter/sfm_data_dance.json',  # _global_oz
                        help="the directory which contains the pictures set.")
    parser.add_argument('--reconstructor', type=str, default='MvgMvsPipeline.py',
                        help="the directory which contains the reconstructor python script.")
    parser.add_argument('--remote', type=str, default='no',
                        help="get images from remote user.")
    parser.add_argument('--image_number', type=str, default="7",
                        help="number of cameras")
    parser.add_argument('--compress', type=float, default=100,
                        help="image compress quality.")
    parser.add_argument('--bg', type=int, default=1, required=True,
                        help="do background subtraction.")
    parser.add_argument('--bg_new', type=int, default=1,
                        help="update background.")
    parser.add_argument('--worker', type=int, default=1,
                        help="number of servers.")
    parser.add_argument('--profile', type=int, default=0,
                        help="number of servers.")
    parser.add_argument('--resolution', type=float, default=1.0,
                        help="the directory which contains the reconstructor python script.")
    parser.add_argument('--max_r', type=str, default="1920",
                        help="the directory which contains the reconstructor python script.")
    parser.add_argument('--n', type=str, default="7",
                        help="the directory which contains the reconstructor python script.")
    parser.add_argument('--edge_fg', type=str, default="None",
                        help="the directory which contains the reconstructor python script.")
    parser.add_argument('--edge_bg', type=str, default="None",
                        help="the directory which contains the reconstructor python script.")
    parser.add_argument('--edge', type=str, default="None",
                        help="the directory which contains the reconstructor python script.")
    parser.add_argument('--local_fg', type=str, default="None",
                        help="the directory which contains the reconstructor python script.")
    parser.add_argument('--local_bg', type=str, default="None",
                        help="the directory which contains the reconstructor python script.")
    parser.add_argument('--local', type=str, default="None",
                        help="the directory which contains the reconstructor python script.")
    parser.add_argument('--new_cfg', type=str, default="0",
                        help="the directory which contains the reconstructor python script.")
    ###################################
    args = parser.parse_args()

    args.bg_new = args.bg_new if args.bg == 1 else 0

    args.local = args.local if args.bg == 0 else 'None'
    args.edge = args.edge if args.bg == 0 and args.worker == 2 else 'None'

    args.edge_fg = args.edge_fg if args.bg == 1 and args.worker == 2 else 'None'
    args.edge_bg = args.edge_bg if args.bg == 1 and args.worker == 2 and args.bg_new == 1 else 'None'

    args.local_fg = args.local_fg if args.bg == 1 else 'None'
    args.local_bg = args.local_bg if args.bg == 1 and args.bg_new == 1 else 'None'
    #########################################
    Path(args.fg_dir).mkdir(parents=True, exist_ok=True)
    Path(args.bg_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_collect_dir).mkdir(parents=True, exist_ok=True)

    return args


def initial_task_setting(args, task_setting_file="task.json"):
    with open(task_setting_file, "r") as jsonFile:
        task = json.load(jsonFile)

    if args.worker == 2:
        task_setting = task["two"]

        for items in task_setting["fg_task_setting"]:
            if items["type"] == "edge":
                items["task"] = args.edge_fg

            if items["type"] == "local":
                items["task"] = args.local_fg

        for items in task_setting["bg_task_setting"]:
            if items["type"] == "edge":
                items["task"] = args.edge_bg

            if items["type"] == "local":
                items["task"] = args.local_bg

        for items in task_setting["task_setting"]:
            if items["type"] == "edge":
                items["task"] = args.edge
            if items["type"] == "local":
                items["task"] = args.local

    else:
        task_setting = task["one"]
        task_setting["fg_task_setting"][0]["task"] = args.local_fg
        task_setting["bg_task_setting"][0]["task"] = args.local_bg
        task_setting["task_setting"][0]["task"] = args.local

    if args.bg == 1:
        print(logger() + "Selected foreground task setting = ",
              f"{bcolors.OKGREEN}" + json.dumps(task_setting["fg_task_setting"], indent=4) + f"{bcolors.ENDC}")
        if args.bg_new == 1:
            print(logger() + "Selected background task setting =",
                  f"{bcolors.OKGREEN}" + json.dumps(task_setting["bg_task_setting"], indent=4) + f"{bcolors.ENDC}")
    else:
        print(logger() + "Selected task setting =",
              f"{bcolors.OKGREEN}" + json.dumps(task_setting["task_setting"], indent=4) + f"{bcolors.ENDC}")

    print(logger() + '#' * 100)

    return task_setting


def create_new_dir(args, timestamp):
    args.output_dir = "data/gold_results_" + args.max_r

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    result_dir = "no_background_sub_" + args.max_r if args.bg != 1 else "with_background_sub_" + args.max_r

    Path("data/" + result_dir).mkdir(parents=True, exist_ok=True)

    str_timestamp = str(timestamp).zfill(5)

    collect_dir = os.path.join(args.data_collect_dir, str_timestamp)
    output_dir = os.path.join(args.output_dir, str_timestamp + "_output")
    fg_dir = os.path.join(args.fg_dir, str_timestamp)
    bg_dir = os.path.join(args.bg_dir, str_timestamp)

    try:
        shutil.rmtree(collect_dir)
        shutil.rmtree(output_dir)
    except:
        pass

    Path(collect_dir).mkdir(parents=True, exist_ok=True)
    Path(fg_dir).mkdir(parents=True, exist_ok=True)
    Path(bg_dir).mkdir(parents=True, exist_ok=True)

    return str_timestamp, collect_dir, output_dir, fg_dir, bg_dir, result_dir