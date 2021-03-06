from utils.utils import *
from dataset.vocdataset import VOC_CLASSES
from dataset.cocodataset import COCO_CLASSES
from dataset.swimdataset import SWIM_CLASSES
from dataset.data_augment import ValTransform
from utils.vis_utils import vis

import os
import sys
import argparse
import yaml
import cv2

import torch
from torch.autograd import Variable
import time

######## unlimit the resource in some dockers or cloud machines ####### 
#import resource
#rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def demo(args, video_pth, video_name):
    """
    YOLOv3 demo. See README for details.
    """
    # Parse config settings
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    #print("successfully loaded config file: ", cfg)

    backbone=cfg['MODEL']['BACKBONE']
    test_size = (args.test_size,args.test_size)

    if args.dataset == 'COCO':
        class_names = COCO_CLASSES
        num_class=80
    if args.dataset == 'VOC':
        class_names = VOC_CLASSES
        num_class=20
    if args.dataset == 'SWIM':
        class_names = SWIM_CLASSES
        num_class=1
    else:
        raise Exception("Only support COCO, VOC ang SWIM model now!")

    # Initiate model
    if args.asff:
        if backbone == 'mobile':
            from models.yolov3_mobilev2 import YOLOv3
            #print("For mobilenet, we currently don't support dropblock, rfb and FeatureAdaption")
        else:
            from models.yolov3_asff import YOLOv3
        #print('Training YOLOv3 with ASFF!')
        model = YOLOv3(num_classes = num_class, rfb=args.rfb, asff=args.asff)
    else:
        if backbone == 'mobile':
            from models.yolov3_mobilev2 import YOLOv3
        else:
            from models.yolov3_baseline import YOLOv3
        #print('Training YOLOv3 strong baseline!')
        model = YOLOv3(num_classes = num_class, rfb=args.rfb)


    if args.checkpoint:
        #print("loading pytorch ckpt...", args.checkpoint)
        cpu_device = torch.device("cpu")
        ckpt = torch.load(args.checkpoint, map_location=cpu_device)
        #model.load_state_dict(ckpt,strict=False)
        model.load_state_dict(ckpt)
    if cuda:
        
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
        model = model.to(device)

    if args.half:
        model = model.half()

    model = model.eval()
    dtype = torch.float16 if args.half else torch.float32

    #load video
    transform = ValTransform(rgb_means=(0.485, 0.456, 0.406), std=(0.229,0.224,0.225))
    cap = cv2.VideoCapture(video_pth)
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))#4 
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))#3
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vid_writer = cv2.VideoWriter(os.path.join(args.save_path,video_name+'.avi'), fourcc, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            ori_frame = frame.copy()
            frame_input, _ = transform(frame, None, test_size)
            if cuda:
                frame_input = frame_input.to(device)

            frame_input = Variable(frame_input.type(dtype).unsqueeze(0))
            outputs= model(frame_input)
            outputs = postprocess(outputs, num_class, 0.01, 0.65)
            if type(outputs) != type([None]):
                outputs = outputs[0].cpu().data
                bboxes = outputs[:, 0:4]
                bboxes[:, 0::2] *= width / test_size[0]
                bboxes[:, 1::2] *= height / test_size[1]
                #bboxes[:, 2] = bboxes[:,2] - bboxes[:,0]
                #bboxes[:, 3] = bboxes[:,3] - bboxes[:,1]
                cls = outputs[:, 6]
                scores = outputs[:, 4]* outputs[:,5]

                pred_frame=vis(ori_frame, bboxes.numpy(), scores.numpy(), cls.numpy(), conf=0.6, class_names=class_names)
            else:
                pred_frame = ori_frame
            cv2.namedWindow("Detection",0)
            cv2.resizeWindow("enhanced", 720, 640)
            cv2.imshow('Detection', pred_frame)
            vid_writer.write(pred_frame)
        else:
            break
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/yolov3_baseline.cfg',
                        help='config file. see readme')
    parser.add_argument('-d', '--dataset', type=str, default='SWIM')
    parser.add_argument('-v', '--video', type=str, default='./example/gray',)
    parser.add_argument('-save_path', type=str, default='./result',)
    parser.add_argument('-c', '--checkpoint', type=str,
                        help='pytorch checkpoint file path')
    parser.add_argument('-s', '--test_size', type=int, default=416)
    parser.add_argument('--half', dest='half', action='store_true', default=False,
                        help='FP16 training')
    parser.add_argument('--rfb', dest='rfb', action='store_true', default=False,
                        help='Use rfb block')
    parser.add_argument('--asff', dest='asff', action='store_true', default=False,
                        help='Use ASFF module for yolov3')
    parser.add_argument('--use_cuda', type=bool, default=True)
    args = parser.parse_args()
    print(args)
    cuda = torch.cuda.is_available() and args.use_cuda
    if cuda:
        print("using cuda")
    
    for video_pth in os.listdir(args.video):
        video_name = video_pth.split('.')[0]
        demo(args, os.path.join(args.video, video_pth), video_name)
        print("finished process video {}".format(video_pth))
    sys.exit(0)