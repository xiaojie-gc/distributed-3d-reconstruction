import argparse
import time
from _thread import start_new_thread
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import torch
from utils.augmentations import letterbox
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords
from utils.plots import colors
from utils.torch_utils import select_device
import cv2

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
         'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
         'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
         'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
         'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
         'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
         'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
         'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
         'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


class Server:
    def __init__(self, opt):
        self.opt = opt

        # Initialize
        self.device = select_device(opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(opt.weights, map_location=self.device)  # load FP32 model
        self.stride = int(model.stride.max())  # model stride
        self.imgsz = check_img_size(opt.img_size, s=self.stride)  # check img_size
        if self.half:
            model.half()  # to FP16

        # Run inference
        if self.device.type != 'cpu':
            model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(model.parameters())))  # run once

        self.model = model

        self.inx = 0


def plot_one_box2(advancement, mask, mask2,  x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness

    x1, y1, x2, y2 = int(x[0]), int(x[1]), int(x[2]), int(x[3])

    c1_fg, c2_fg = (max(0, x1 - advancement), max(0, y1 - advancement)), (min(x2 + advancement, im.shape[1]), min(y2 + advancement, im.shape[0]))

    c1_bg, c2_bg = (x1, y1), (x2, y2)

    #mask = np.zeros(im.shape, dtype=np.uint8)
    cv2.rectangle(mask, c1_fg, c2_fg, (255, 255, 255), -1)

    cv2.rectangle(mask2, c1_bg, c2_bg, (255, 255, 255), -1)

    #cv2.rectangle(im, c1_bg, c2_bg, color, thickness=tl, lineType=cv2.LINE_AA)
    # return result, mask2


def img_postprocessing(fg_dir, bg_dir, pred, img, im0s, advancement, target_classes):
    box = []
    fg_ratio = []
    found = 0
    for i, det in enumerate(pred):  # detections per image

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

            mask = np.zeros(im0s.shape, dtype=np.uint8)
            mask2 = np.zeros(im0s.shape, dtype=np.uint8)

            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                # label = f'{names[c]} {conf:.2f}'
                # print(names[c])
                if target_classes is not None:
                    if names[c] in target_classes: #
                        plot_one_box2(advancement, mask, mask2, xyxy, im0s, label=False, color=colors(c, True),
                                                    line_thickness=3)
                        found += 1
                        box.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                else:
                    plot_one_box2(advancement, mask, mask2, xyxy, im0s, label=False, color=colors(c, True),
                                  line_thickness=3)
                    found += 1
                    box.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])

            #cv2.imwrite(fg_dir, im0s)

            result = cv2.bitwise_and(im0s, mask)
            result[mask == 0] = 0

            cv2.imwrite(fg_dir, result)

            bg = cv2.bitwise_not(mask2)
            bg = cv2.bitwise_and(im0s, bg)
            cv2.imwrite(bg_dir, bg)

            sought = [0, 0, 0]
            ratio = round((1920*1080 - np.count_nonzero(np.all(mask == sought, axis=2)))/(1920*1080), 2)

    # msa.append(box)
    return [box, ratio, found]


@torch.no_grad()
def detect(items):
    advancement, im_file, fg_dir, bg_dir, model, imgsz, stride, half, device, conf_thres, target_classes, opt = items
    im0s = cv2.imread(im_file)  # BGR
    img = letterbox(im0s, imgsz, stride=stride)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)

    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                               max_det=opt.max_det)

    # start_new_thread(img_postprocessing, (fg_dir, bg_dir, pred, img, im0s, advancement, msa))
    return img_postprocessing(fg_dir, bg_dir, pred, img, im0s, advancement, target_classes)
    # return msa, found


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.05, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--line-thickness', default=5, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')

    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    s = Server(opt)

    msa = []

    start = time.time()

    items = []

    adc = 0

    items.append((adc, "data/000.jpg", "data/000_fg.jpg", "data/000_bg.jpg", s.model, s.imgsz, s.stride, s.half,
                 s.device,
                 s.opt))
    items.append((adc, "data/001.jpg", "data/001_fg.jpg", "data/001_bg.jpg", s.model, s.imgsz, s.stride, s.half,
                 s.device,
                 s.opt))
    items.append((adc, "data/002.jpg", "data/002_fg.jpg", "data/002_bg.jpg", s.model, s.imgsz, s.stride, s.half,
                 s.device,
                 s.opt))
    items.append((adc, "data/003.jpg", "data/003_fg.jpg", "data/003_bg.jpg", s.model, s.imgsz, s.stride, s.half,
                 s.device,
                 s.opt))
    items.append((adc, "data/004.jpg", "data/004_fg.jpg", "data/004_bg.jpg", s.model, s.imgsz, s.stride, s.half,
                 s.device,
                 s.opt))
    items.append((adc, "data/005.jpg", "data/005_fg.jpg", "data/005_bg.jpg", s.model, s.imgsz, s.stride, s.half,
                 s.device,
                 s.opt))
    items.append((adc, "data/006.jpg", "data/006_fg.jpg", "data/006_bg.jpg", s.model, s.imgsz, s.stride, s.half,
                 s.device,
                 s.opt))

    with ThreadPoolExecutor(max_workers=1) as executor: #1
        results = executor.map(detect, items)

    for result in results:
        msa.append(result)

    while len(msa) < 7:
        pass

    print(time.time() - start)
    print(msa)

