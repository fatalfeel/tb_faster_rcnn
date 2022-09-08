import argparse
import os
import time
import torch
from dataset.baseobject import DatasetBase
from backbone.basenet import BackboneBase
from config.eval_config import EvalConfig as Config
from evaluator import Evaluator
from logger import Logger as Log
from model import Model

def str2bool(b_str):
    if b_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif b_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',                        type=str,       default='voc2007',          help='name of dataset')
parser.add_argument('--backbone',                       type=str,       default='resnext101_32x8d', help='resnet18, resnet50, resnet101, resnext101_32x8d')
parser.add_argument('--data_dir',                       type=str,       default='./data',           help='path to data directory')
parser.add_argument('--checkpoint',                     type=str,       default='./checkpoint',     help='path to checkpoint')
parser.add_argument('--image_min_side',                 type=float,                                 help='default: {:g}'.format(Config.IMAGE_MIN_SIDE))
parser.add_argument('--image_max_side',                 type=float,                                 help='default: {:g}'.format(Config.IMAGE_MAX_SIDE))
parser.add_argument('--anchor_ratios',                  type=str,                                   help='default: "{!s}"'.format(Config.ANCHOR_RATIOS))
parser.add_argument('--anchor_sizes',                   type=str,                                   help='default: "{!s}"'.format(Config.ANCHOR_SIZES))
#parser.add_argument('--pooler_mode',                   type=str,       choices=Pooler.OPTIONS,     help='default: {.value:s}'.format(Config.POOLER_MODE))
parser.add_argument('--rpn_pre_nms_top_n',              type=int,                                   help='default: {:d}'.format(Config.RPN_PRE_NMS_TOP_N))
parser.add_argument('--rpn_post_nms_top_n',             type=int,                                   help='default: {:d}'.format(Config.RPN_POST_NMS_TOP_N))
parser.add_argument('--batch_size',                     type=int,                                   help='default: {:g}'.format(Config.BATCH_SIZE))
parser.add_argument('--output',                         type=str,       default='./output',         help='path to output result image')
parser.add_argument('--cuda',                           type=str2bool,  default=False)
args = parser.parse_args()

def _eval(path_to_results_dir: str):
    device      = torch.device("cuda" if args.cuda else "cpu")
    kwargs      = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

    dataset     = DatasetBase.from_name(args.dataset)(args.data_dir, DatasetBase.Mode.EVAL, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)
    evaluator   = Evaluator(dataset, Config.BATCH_SIZE, args.data_dir, path_to_results_dir, device, kwargs)

    Log.i('Found {:d} samples'.format(len(dataset)))

    backbone    = BackboneBase.from_name(args.backbone)(pretrained=True)
    model       = Model(backbone,
                        dataset.num_classes(),
                        #pooler_mode=Config.POOLER_MODE,
                        anchor_ratios     = Config.ANCHOR_RATIOS,
                        anchor_sizes      = Config.ANCHOR_SIZES,
                        rpn_pre_nms_top_n = Config.RPN_PRE_NMS_TOP_N,
                        rpn_post_nms_top_n= Config.RPN_POST_NMS_TOP_N).to(device)

    model.load(args.checkpoint)

    Log.i('Start evaluating with 1 GPU (1 batch per GPU)')
    mean_ap, detail = evaluator.evaluate(model)
    Log.i('Done')

    Log.i('mean AP = {:.4f}'.format(mean_ap))
    Log.i('\n' + detail)

if __name__ == '__main__':
#def main():
    '''parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--dataset', type=str, choices=DatasetBase.OPTIONS, required=True, help='name of dataset')
    parser.add_argument('-b', '--backbone', type=str, choices=BackboneBase.OPTIONS, required=True, help='name of backbone model')
    parser.add_argument('-d', '--data_dir', type=str, default='./data', help='path to data directory')
    parser.add_argument('--image_min_side', type=float, help='default: {:g}'.format(Config.IMAGE_MIN_SIDE))
    parser.add_argument('--image_max_side', type=float, help='default: {:g}'.format(Config.IMAGE_MAX_SIDE))
    parser.add_argument('--anchor_ratios', type=str, help='default: "{!s}"'.format(Config.ANCHOR_RATIOS))
    parser.add_argument('--anchor_sizes', type=str, help='default: "{!s}"'.format(Config.ANCHOR_SIZES))
    parser.add_argument('--pooler_mode', type=str, choices=Pooler.OPTIONS, help='default: {.value:s}'.format(Config.POOLER_MODE))
    parser.add_argument('--rpn_pre_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_PRE_NMS_TOP_N))
    parser.add_argument('--rpn_post_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_POST_NMS_TOP_N))
    parser.add_argument('checkpoint', type=str, help='path to evaluating checkpoint')
    args = parser.parse_args()'''

    path_to_results_dir = os.path.join(args.output,'{:s}'.format(time.strftime('%Y%m%d%H%M%S')))
    os.makedirs(path_to_results_dir)

    Config.setup(image_min_side=args.image_min_side,
                 image_max_side=args.image_max_side,
                 anchor_ratios=args.anchor_ratios,
                 anchor_sizes=args.anchor_sizes,
                 #pooler_mode=args.pooler_mode,
                 rpn_pre_nms_top_n=args.rpn_pre_nms_top_n,
                 rpn_post_nms_top_n=args.rpn_post_nms_top_n)

    Log.initialize(os.path.join(path_to_results_dir, 'eval.log'))
    Log.i('Arguments:')
    for k, v in vars(args).items():
        Log.i(f'\t{k} = {v}')
    Log.i(Config.describe())

    _eval(path_to_results_dir)
#main()
