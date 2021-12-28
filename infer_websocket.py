import argparse
import asyncio
import json
import numpy as np
import torch
import websockets
from dataset.baseobject import DatasetBase
from backbone.basenet import BackboneBase
from PIL import Image
from bbox import BBox
from config.eval_config import EvalConfig as Config
from model import Model
#from roi.pooler import Pooler

def str2bool(b_str):
    if b_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif b_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',                type=str,   default='voc2007',      help='name of dataset')
parser.add_argument('--backbone',               type=str,   default='resnet101',    help='resnet18, resnet50, resnet101')
parser.add_argument('--checkpoint_dir',         type=str,   default='./checkpoint', help='path to checkpoint')
parser.add_argument('--probability_threshold',  type=float, default=0.5,            help='threshold of detection probability')
parser.add_argument('--image_min_side', type=float, help='default: {:g}'.format(Config.IMAGE_MIN_SIDE))
parser.add_argument('--image_max_side', type=float, help='default: {:g}'.format(Config.IMAGE_MAX_SIDE))
parser.add_argument('--anchor_ratios',  type=str, help='default: "{!s}"'.format(Config.ANCHOR_RATIOS))
parser.add_argument('--anchor_sizes',   type=str, help='default: "{!s}"'.format(Config.ANCHOR_SIZES))
#parser.add_argument('--pooler_mode', type=str, choices=Pooler.OPTIONS, help='default: {.value:s}'.format(Config.POOLER_MODE))
parser.add_argument('--rpn_pre_nms_top_n',  type=int, help='default: {:d}'.format(Config.RPN_PRE_NMS_TOP_N))
parser.add_argument('--rpn_post_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_POST_NMS_TOP_N))
parser.add_argument('--anchor_smooth_l1_loss_beta', type=float, help='default: {:g}'.format(Config.ANCHOR_SMOOTH_L1_LOSS_BETA))
parser.add_argument('--proposal_smooth_l1_loss_beta', type=float, help='default: {:g}'.format(Config.PROPOSAL_SMOOTH_L1_LOSS_BETA))
parser.add_argument('--batch_size', type=int, help='default: {:g}'.format(Config.BATCH_SIZE))
parser.add_argument('--learning_rate', type=float, help='default: {:g}'.format(Config.LEARNING_RATE))
#parser.add_argument('--momentum', type=float, help='default: {:g}'.format(Config.MOMENTUM))
#parser.add_argument('--weight_decay', type=float, help='default: {:g}'.format(Config.WEIGHT_DECAY))
#parser.add_argument('--step_lr_sizes', type=str, help='default: {!s}'.format(Config.STEP_LR_SIZES))
#parser.add_argument('--step_lr_gamma', type=float, help='default: {:g}'.format(Config.STEP_LR_GAMMA))
#parser.add_argument('--warm_up_factor', type=float, help='default: {:g}'.format(Config.WARM_UP_FACTOR))
#parser.add_argument('--warm_up_num_iters', type=int, help='default: {:d}'.format(Config.WARM_UP_NUM_ITERS))
parser.add_argument('--cuda', default=False, type=str2bool)
args = parser.parse_args()

device  = torch.device("cuda" if args.cuda else "cpu")

def _infer_websocket(dataset_name: str, backbone_name: str, prob_thresh: float):
    dataset_class = DatasetBase.from_name(dataset_name)
    backbone = BackboneBase.from_name(backbone_name)(pretrained=True)
    model = Model(backbone,
                  dataset_class.num_classes(),
                  #pooler_mode=Config.POOLER_MODE,
                  anchor_ratios=Config.ANCHOR_RATIOS,
                  anchor_sizes=Config.ANCHOR_SIZES,
                  rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N,
                  rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N).to(device)
    model.load(args.checkpoint_dir)

    async def handler(websocket, path):
        print('Connection established:', path)

        with torch.no_grad():
            while True:
                frame = await websocket.recv()
                frame = np.frombuffer(frame, dtype=np.uint8).reshape(480, 640, 3)

                image = Image.fromarray(frame)
                image_tensor, scale = dataset_class.preprocess(image, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)

                detection_bboxes, detection_classes, detection_probs, _ = \
                    model.eval().forward(image_tensor.unsqueeze(dim=0).cuda())
                detection_bboxes /= scale

                kept_indices = detection_probs > prob_thresh
                detection_bboxes = detection_bboxes[kept_indices]
                detection_classes = detection_classes[kept_indices]
                detection_probs = detection_probs[kept_indices]

                message = []

                for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(), detection_probs.tolist()):
                    bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
                    category = dataset_class.LABEL_TO_CATEGORY_DICT[cls]

                    message.append({'left': int(bbox.left),
                                    'top': int(bbox.top),
                                    'right': int(bbox.right),
                                    'bottom': int(bbox.bottom),
                                    'category': category})

                message = json.dumps(message)
                await websocket.send(message)

    server = websockets.serve(handler, host='*', port=8765, max_size=2 ** 32, compression=None)
    asyncio.get_event_loop().run_until_complete(server)
    print('Service is ready. Please navigate to http://127.0.0.1:8000/')
    asyncio.get_event_loop().run_forever()


if __name__ == '__main__':
#def main():
    '''parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--dataset', type=str, choices=DatasetBase.OPTIONS, required=True, help='name of dataset')
    parser.add_argument('-b', '--backbone', type=str, choices=BackboneBase.OPTIONS, required=True, help='name of backbone model')
    parser.add_argument('-c', '--checkpoint_dir', type=str, required=True, help='path to checkpoint')
    parser.add_argument('-p', '--probability_threshold', type=float, default=0.6, help='threshold of detection probability')
    parser.add_argument('--image_min_side', type=float, help='default: {:g}'.format(Config.IMAGE_MIN_SIDE))
    parser.add_argument('--image_max_side', type=float, help='default: {:g}'.format(Config.IMAGE_MAX_SIDE))
    parser.add_argument('--anchor_ratios', type=str, help='default: "{!s}"'.format(Config.ANCHOR_RATIOS))
    parser.add_argument('--anchor_sizes', type=str, help='default: "{!s}"'.format(Config.ANCHOR_SIZES))
    #parser.add_argument('--pooler_mode', type=str, choices=Pooler.OPTIONS, help='default: {.value:s}'.format(Config.POOLER_MODE))
    parser.add_argument('--rpn_pre_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_PRE_NMS_TOP_N))
    parser.add_argument('--rpn_post_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_POST_NMS_TOP_N))
    args = parser.parse_args()'''

    dataset_name = args.dataset
    backbone_name = args.backbone
    #path_to_checkpoint = args.checkpoint_dir
    prob_thresh = args.probability_threshold

    Config.setup(image_min_side=args.image_min_side,
                 image_max_side=args.image_max_side,
                 anchor_ratios=args.anchor_ratios,
                 anchor_sizes=args.anchor_sizes,
                 #pooler_mode=args.pooler_mode,
                 rpn_pre_nms_top_n=args.rpn_pre_nms_top_n,
                 rpn_post_nms_top_n=args.rpn_post_nms_top_n)

    print('Arguments:')
    for k, v in vars(args).items():
        print(f'\t{k} = {v}')
    print(Config.describe())

    _infer_websocket(dataset_name, backbone_name, prob_thresh)

#main()
