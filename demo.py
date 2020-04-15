from utils.utils import *
from dataset.vocdataset import VOC_CLASSES
from dataset.cocodataset import COCO_CLASSES
from dataset.swimdataset import SWIM_CLASSES
from dataset.data_augment import ValTransform
from utils.vis_utils import vis

import os
import glob
import sys
import argparse
import yaml
import cv2
cv2.setNumThreads(0)

import torch
from torch.autograd import Variable
import time

######## unlimit the resource in some dockers or cloud machines ####### 
#import resource
#rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

def demo(args, img):
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
        raise Exception("Only support COCO or VOC model now!")

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
        #print("using cuda")
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
        model = model.to(device)

    if args.half:
        model = model.half()

    model = model.eval()
    dtype = torch.float16 if args.half else torch.float32

    #load img
    transform = ValTransform(rgb_means=(0.485, 0.456, 0.406), std=(0.229,0.224,0.225))
    #im = cv2.imread(args.img)
    im = cv2.imread(os.path.join(args.img, img))
    height, width, _ = im.shape
    ori_im = im.copy()
    im_input, _ = transform(im, None, test_size)
    if cuda:
        im_input = im_input.to(device)

    im_input = Variable(im_input.type(dtype).unsqueeze(0))
    outputs= model(im_input) #xc,yc, w, h
    outputs = postprocess(outputs, num_class, 0.1, 0.65)
    outputs = outputs[0].cpu().data

    bboxes = outputs[:, 0:4] #x1, y1, x2, y2
    bboxes[:, 0::2] *= width / test_size[0] #rescale
    bboxes[:, 1::2] *= height / test_size[1] #rescale
    #bboxes[:, 2] = bboxes[:,2] - bboxes[:,0] # w
    #bboxes[:, 3] = bboxes[:,3] - bboxes[:,1] # h
    cls = outputs[:, 6]
    scores = outputs[:, 4]* outputs[:,5]

    pred_im=vis(ori_im, bboxes.numpy(), scores.numpy(), cls.numpy(), conf=0.6, class_names=class_names)
    cv2.imshow('Detection', pred_im)
    cv2.imwrite(os.path.join(args.saved, img), pred_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return bboxes.numpy(), cls.numpy(), scores.numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/yolov3_baseline.cfg',
                        help='config file. see readme')
    parser.add_argument('-d', '--dataset', type=str, default='SWIM')
    parser.add_argument('-i', '--img', type=str, default='./example/swim',)
    parser.add_argument('-saved', type=str, default='./result/swim')
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
    for img in os.listdir(args.img):
        bboxes, cls, scores = demo(args, img)
        print("bboxes: {}, cls: {}, scores: {}".format(bboxes, cls, scores))
    sys.exit(0)