import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import socket
import selectors
import time
import types
import math
import numpy as np
import bezier
import matplotlib.pyplot as plt

DEBUG_MODE = True

cooldown = 0.01
WALK_SENS = 0.85
ROT_SENS = 0.75
HEIGHT_SENS = 0.9
ROLL_SENS = 0.6

# this value changes the amount the dog rotates, and therefore the length of the path.
NORMALIZATION_CONSTANT = 1280 * 2
CAM_WIDTH = 1280 #320  # 1280
CAM_HEIGHT = 720 #240  # 720

TCP_IP = '10.40.7.255'
TCP_PORT = 5005
BUFFER_SIZE = 128

KEY = b'#'
SEP = b'@'

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def start_connection():
    global ds
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setblocking(False)
    s.connect_ex((TCP_IP, TCP_PORT))
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    ds = types.SimpleNamespace(connid=1, msg_total=0, recv_total=0, messages=list(), outb=b'')
    sel.register(s, events, data=ds)
    print('connection started!')


def service_connection(key, mask):
    sock = key.fileobj
    data = key.data
    if mask & selectors.EVENT_READ:
        recv_data = sock.recv(128)  # Should be ready to read
        if recv_data:
            # print('received', repr(recv_data), 'from connection', data.connid)
            data.recv_total += len(recv_data)
        if not recv_data or data.recv_total == data.msg_total:
            print('closing connection', data.connid)
            sel.unregister(sock)
            sock.close()
    if mask & selectors.EVENT_WRITE:
        if not data.outb and data.messages:
            # data.outb = b''
            data.outb = data.messages.pop(0)
        if data.outb:
            print('sending', repr(data.outb), 'to connection', data.connid)
            sent = sock.send(data.outb)  # Should be ready to write
            data.outb = data.outb[sent:]


def send_commands(inputs):
    msg = b'rx' + KEY + bytes(str(inputs['rx']), 'utf-8') \
          + SEP + b'ry' + KEY + bytes(str(inputs['ry']), 'utf-8') \
          + SEP + b'lx' + KEY + bytes(str(inputs['lx']), 'utf-8') \
          + SEP + b'ly' + KEY + bytes(str(inputs['ly']), 'utf-8') \
          + SEP + b'dpadx' + KEY + bytes(str(inputs['dpadx']), 'utf-8') \
          + SEP + b'dpady' + KEY + bytes(str(inputs['dpady']), 'utf-8') \
          + SEP + b'trot' + KEY + bytes(str(inputs['trot']), 'utf-8') \
          + SEP

    # print(msg)
    if len(ds.messages) < 1:
        ds.messages.append(msg)
    else:
        ds.messages[0] = msg


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def plot_bez(curve, frame):
    RES = 64.
    RES_I = int(RES)

    pts = curve.nodes

    xs = np.zeros((RES_I, 1))
    ys = np.zeros((RES_I, 1))

    for i in range(RES_I):
        x, y = curve.evaluate(i / RES)
        xs[i] = x
        ys[i] = y

    plt.plot(xs, ys)
    plt.plot(pts[0][:], pts[1][:], '-')

    plt.xlabel('x coord')
    plt.ylabel('distance')

    plt.xlim((-10, CAM_WIDTH + 10))
    plt.ylim((-0.2, 1.2))
    plt.savefig('pics/result' + str(frame) + '.png')
    plt.close()


def y_coord_sort(e):
    x, y = e
    return y


def detect(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA
    last_time = time.time()

    # Load model
    model = torch.load(weights, map_location=device)[
        'model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'
    print('starting predictions...')

    # static vars
    time_total_start = 0

    curve_goal = None

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):

        if curve_goal is not None:  # CURVE ACTION
            # eval curve
            total_change = np.array([0], dtype='float64')
            total_time = 10
            time_elapsed = time_synchronized() - time_total_start

            x_start = curve_goal.evaluate(time_elapsed/total_time)[0]
            for a in range(int(time_elapsed * 32), int((time_elapsed + 0.25) * 32)):
                if time_elapsed + 0.25 > total_time:
                    break
                total_change += curve_goal.evaluate(a / (total_time * 32))[0] - x_start
                # print(total_change)
                # if total_change? > 9999999:
                #     print('broke at', total_change, '[max 999999]')

            total_change /= (time_elapsed + 0.5) / total_time

            print((total_change / NORMALIZATION_CONSTANT)[0])
            # print('time', time_elapsed/total_time, '\n')

            rx = max(min(total_change / NORMALIZATION_CONSTANT, [1]), [-1])[0]

            vals['rx'] = round(rx, 5)
            vals['ly'] = 1
            vals['trot'] = 1
            # print(vals['rx'])

            if time_synchronized() - time_total_start > total_time:
                curve_goal = None
                time_total_start = 0

                vals['trot'] = 0
                vals['ly'] = 0
                # sys.exit()
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(
                pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                s += '%gx%g ' % img.shape[2:]  # print string
                save_path = str(Path(out) / Path(p).name)

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    bbox_xywh = []
                    confs = []

                    # Adapt detections to deep sort input format
                    for *xyxy, conf, cls in det:
                        x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                        obj = [x_c, y_c, bbox_w, bbox_h]
                        bbox_xywh.append(obj)
                        confs.append([conf.item()])

                    xywhs = torch.Tensor(bbox_xywh)
                    confss = torch.Tensor(confs)

                    # Pass detections to deepsort
                    outputs = deepsort.update(xywhs, confss, im0)

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        draw_boxes(im0, bbox_xyxy, identities)

                    # Write MOT compliant results to file
                    if save_txt and len(outputs) != 0:
                        bezier_points = np.zeros((2 * len(outputs), 2))
                        idx = 0

                        for j, output in enumerate(outputs):
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2]
                            bbox_h = output[3]
                            identity = output[-1]
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
                            # calculating rotation and movement for dog!
                            # gotta use vals['...']
                            # dim CAM_WIDTH x CAM_HEIGHT
                            # rotation calculation
                            middle_x = (bbox_left + bbox_w) / 2

                            # translation calculation
                            percent_filled_y = (bbox_h - bbox_top) / CAM_HEIGHT
                            percent_filled_y *= 100

                            # GENERATE TOLERANCE #
                            max_width_tolerance = 0.375 * (bbox_w - bbox_left)
                            left_bound = bbox_left - max_width_tolerance
                            right_bound = bbox_w + max_width_tolerance

                            # GENERATE DISTANCE #
                            distance_rel = math.e**(-percent_filled_y/30)
                            # print('left, middle, right', left_bound, middle_x, right_bound)
                            # print('dist', distance_rel, 'pct', percent_filled_y)

                            # GENERATE POINTS #
                            if percent_filled_y < 1.:
                                bezier_points[idx][0] = -1
                                bezier_points[idx][1] = -1
                                bezier_points[idx + 1][0] = -1
                                bezier_points[idx + 1][1] = -1
                            else:
                                if idx > 1: # relative to last box
                                    midpoint = bezier_points[idx - 1][0] # exit point of last node
                                    bezier_points[idx][0] = left_bound if middle_x >= midpoint else right_bound
                                    bezier_points[idx][1] = distance_rel - 0.005
                                    bezier_points[idx + 1][0] = left_bound if middle_x >= midpoint else right_bound
                                    bezier_points[idx + 1][1] = distance_rel + 0.005
                                else: # rel to middle
                                    bezier_points[idx][0] = left_bound if middle_x >= (CAM_WIDTH / 2) else right_bound
                                    bezier_points[idx][1] = distance_rel - 0.005
                                    bezier_points[idx + 1][0] = left_bound if middle_x >= (CAM_WIDTH / 2) else right_bound
                                    bezier_points[idx + 1][1] = distance_rel + 0.005

                            idx += 2

                        if cv2.waitKey(1) == ord('f'):
                            points = list()
                            skipped_boxes = list()
                            skip_idx = -1
                            for a in range(bezier_points.shape[0]):
                                x = bezier_points[a][0]
                                y = bezier_points[a][1]

                                if bezier_points.shape[0] > a + 1 and abs(y - bezier_points[a+1][1]) < .001:
                                    skip_idx = a + 1

                                if x < 0 or y < 0:
                                    continue

                                if bezier_points.shape[0] > a + 3 and (a != skip_idx) and abs(x - bezier_points[a+2][0]) > CAM_WIDTH * 0.2:  # threshold for ignorance
                                    # print('skipping', a + 2)
                                    skipped_boxes.append(a + 2)
                                    skipped_boxes.append(a + 3)

                            for skipped_idx in skipped_boxes:
                                bezier_points[skipped_idx][0] = -1
                                bezier_points[skipped_idx][1] = -1

                            for a in range(bezier_points.shape[0]):
                                x = bezier_points[a][0]
                                y = bezier_points[a][1]
                                if x < 0 or y < 0:
                                    continue
                                points.append((x, y))

                            far_pt = 0
                            if len(points) > 0:
                                far_pt = points[-1][1] + 1
                            points.append((CAM_WIDTH/2, 0))
                            points.append((CAM_WIDTH/2, far_pt))
                            points.sort(key=y_coord_sort)

                            nodes_curve_norm = np.swapaxes(np.array(points), 1, 0)
                            nodes_curve = np.asfortranarray(nodes_curve_norm)
                            # print(frame_idx, nodes_curve)
                            print('calculating curve on frame', frame_idx)

                            curve = bezier.Curve(nodes_curve, degree=(nodes_curve.shape[1] - 1))

                            # DISPLAY DATA #

                            # x = left/right bound [0,720]
                            # y = distance [1, e**-3]
                            for j, output in enumerate(outputs):
                                bbox_left = output[0]
                                bbox_top = output[1]
                                bbox_w = output[2]
                                bbox_h = output[3]

                                percent_filled_y = (bbox_h - bbox_top) / CAM_HEIGHT
                                percent_filled_y *= 100
                                distance_rel = math.e ** (-percent_filled_y / 30)

                                plt.plot([bbox_left, bbox_w], [distance_rel, distance_rel])
                            plot_bez(curve, frame_idx)

                            # set curve
                            time_total_start = time_synchronized()
                            curve_goal = curve

                            # sys.exit()

                else:
                    deepsort.increment_ages()

                # Print time (inference + NMS)
                # print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results
                if view_img:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    print('saving img!')
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
                    else:
                        print('saving video!')
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(
                                save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                        vid_writer.write(im0)
        if not DEBUG_MODE:
            if time.time() - last_time > cooldown:  # so we dont spam
                send_commands(vals)
                last_time = time.time()
            events = sel.select(timeout=1)
            if events:
                for key, mask in events:
                    service_connection(key, mask)
            if not sel.get_map():
                break


    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/weights/yolov5s.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    # print(args)

    vals = {
        'rx': 0,
        'ry': 0,
        'lx': 0,
        'ly': 0,
        'dpadx': 0,
        'dpady': 0,
        'trot': 0
    }

    # inits
    sel = selectors.DefaultSelector()
    ds = None

    print('starting connection...')
    start_connection()

    with torch.no_grad():
        detect(args)

    sel.close()
