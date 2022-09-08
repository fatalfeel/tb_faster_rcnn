import argparse
import os
import time
import torch
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from collections import deque
from dataset.baseobject import DatasetBase
from backbone.basenet import BackboneBase
from config.train_config import TrainConfig as Config
from logger import Logger as Log
from model import Model
#from roi.pooler import Pooler
#from torch.optim import Optimizer
#from torch.optim.lr_scheduler import MultiStepLR
#from vis_tool import visdom_bbox

def str2bool(b_str):
    if b_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif b_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',                        type=str,       default='voc2007',          help='voc2007, coco2017, voc2007-cat-dog, coco2017-person, coco2017-car, coco2017-animal')
parser.add_argument('--backbone',                       type=str,       default='resnext101_32x8d', help='resnet18, resnet50, resnet101, resnext101_32x8d')
parser.add_argument('--data_dir',                       type=str,       default='./data',           help='path to data directory')
parser.add_argument('--checkpoint_dir',                 type=str,       default='./checkpoint',     help='path to checkpoint')
parser.add_argument('--image_min_side',                 type=float,                                 help='default: {:g}'.format(Config.IMAGE_MIN_SIDE))
parser.add_argument('--image_max_side',                 type=float,                                 help='default: {:g}'.format(Config.IMAGE_MAX_SIDE))
parser.add_argument('--anchor_ratios',                  type=str,                                   help='default: "{!s}"'.format(Config.ANCHOR_RATIOS))
parser.add_argument('--anchor_sizes',                   type=str,                                   help='default: "{!s}"'.format(Config.ANCHOR_SIZES))
#parser.add_argument('--pooler_mode', type=str, choices=Pooler.OPTIONS,                             help='default: {.value:s}'.format(Config.POOLER_MODE))
parser.add_argument('--rpn_pre_nms_top_n',              type=int,                                   help='default: {:d}'.format(Config.RPN_PRE_NMS_TOP_N))
parser.add_argument('--rpn_post_nms_top_n',             type=int,                                   help='default: {:d}'.format(Config.RPN_POST_NMS_TOP_N))
parser.add_argument('--anchor_smooth_l1_loss_beta',     type=float,                                 help='default: {:g}'.format(Config.ANCHOR_SMOOTH_L1_LOSS_BETA))
parser.add_argument('--proposal_smooth_l1_loss_beta',   type=float,                                 help='default: {:g}'.format(Config.PROPOSAL_SMOOTH_L1_LOSS_BETA))
parser.add_argument('--batch_size',                     type=int,                                   help='default: {:g}'.format(Config.BATCH_SIZE))
parser.add_argument('--learning_rate',                  type=float,                                 help='default: {:g}'.format(Config.LEARNING_RATE))
parser.add_argument('--momentum',                       type=float,                                 help='default: {:g}'.format(Config.MOMENTUM))
parser.add_argument('--weight_decay',                   type=float,                                 help='default: {:g}'.format(Config.WEIGHT_DECAY))
#parser.add_argument('--step_lr_sizes',                 type=str,                                   help='default: {!s}'.format(Config.STEP_LR_SIZES))
parser.add_argument('--update_lr_freq',                 type=str,                                   help='default: {!s}'.format(Config.UPDATE_LR_FREQ))
parser.add_argument('--step_lr_gamma',                  type=float,                                 help='default: {:g}'.format(Config.STEP_LR_GAMMA))
#parser.add_argument('--warm_up_factor',                type=float,                                 help='default: {:g}'.format(Config.WARM_UP_FACTOR))
#parser.add_argument('--warm_up_num_iters',             type=int,                                   help='default: {:d}'.format(Config.WARM_UP_NUM_ITERS))
parser.add_argument('--num_steps_to_display',           type=int,                                   help='default: {:d}'.format(Config.NUM_STEPS_TO_DISPLAY))
parser.add_argument('--num_save_epoch_freq',            type=int,                                   help='default: {:d}'.format(Config.NUM_SAVE_EPOCH_FREQ))
parser.add_argument('--num_epoch_to_finish',            type=int,                                   help='default: {:d}'.format(Config.NUM_EPOCH_TO_FINISH))
parser.add_argument('--resume',                         type=str2bool,  default=False,              help='continue training')
parser.add_argument('--cuda',                           type=str2bool,  default=False)
args = parser.parse_args()

def save_MNIST(img, pname):
    #grid = torchvision.utils.make_grid(img)
    #trimg = grid.numpy().transpose(1, 2, 0)
    trimg       = img.transpose(1, 2, 0)
    plt.axis('off')
    plt.imshow(trimg)
    plt.savefig(pname)

'''
class WarmUpMultiStepLR(MultiStepLR):
    def __init__(self,
                 optimizer: Optimizer,
                 milestones: List[int],
                 gamma:float    = 0.1,
                 factor:float   = 0.3333,
                 num_iters:int  = 500,
                 last_epoch:int = -1):
        self.factor = factor
        self.num_iters = num_iters
        super().__init__(optimizer, milestones, gamma, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.num_iters:
            alpha = self.last_epoch / self.num_iters
            factor = (1 - self.factor) * alpha + self.factor
        else:
            factor = 1

        return [lr * factor for lr in super().get_lr()]
'''

#def _train(dataset_name: str, backbone_name: str, path_to_data_dir: str, path_to_checkpoints_dir: str):
def _train():
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

    dataset     = DatasetBase.from_name(args.dataset)(args.data_dir, DatasetBase.Mode.TRAIN, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)
    dataloader  = DataLoader(dataset,
                             batch_size = Config.BATCH_SIZE,
                             sampler    = DatasetBase.NearestRatioRandomSampler(dataset.image_ratios, num_neighbors=Config.BATCH_SIZE),
                             collate_fn = DatasetBase.padding_collate_fn,
                             **kwargs)

    sample_size = len(dataset)
    Log.i('Found {:d} samples'.format(sample_size))

    backbone = BackboneBase.from_name(args.backbone)(pretrained=True)

    #for multi gpu card
    '''model = nn.DataParallel(Model(backbone,
                                  dataset.num_classes(),
                                  #pooler_mode=Config.POOLER_MODE,
                                  anchor_ratios=Config.ANCHOR_RATIOS,
                                  anchor_sizes=Config.ANCHOR_SIZES,
                                  rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N,
                                  rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N,
                                  anchor_smooth_l1_loss_beta=Config.ANCHOR_SMOOTH_L1_LOSS_BETA,
                                  proposal_smooth_l1_loss_beta=Config.PROPOSAL_SMOOTH_L1_LOSS_BETA).to(device))'''
    model = Model(  backbone,
                    dataset.num_classes(),
                    # pooler_mode=Config.POOLER_MODE,
                    anchor_ratios               = Config.ANCHOR_RATIOS,
                    anchor_sizes                = Config.ANCHOR_SIZES,
                    rpn_pre_nms_top_n           = Config.RPN_PRE_NMS_TOP_N,
                    rpn_post_nms_top_n          = Config.RPN_POST_NMS_TOP_N,
                    anchor_smooth_l1_loss_beta  = Config.ANCHOR_SMOOTH_L1_LOSS_BETA,
                    proposal_smooth_l1_loss_beta= Config.PROPOSAL_SMOOTH_L1_LOSS_BETA  ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, momentum=Config.MOMENTUM, weight_decay=Config.WEIGHT_DECAY)
    #scheduler = WarmUpMultiStepLR(optimizer, milestones=Config.STEP_LR_SIZES, gamma=Config.STEP_LR_GAMMA, factor=Config.WARM_UP_FACTOR, num_iters=Config.WARM_UP_NUM_ITERS)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(sample_size/Config.UPDATE_LR_FREQ), gamma=Config.STEP_LR_GAMMA)

    num_steps_to_display    = Config.NUM_STEPS_TO_DISPLAY
    num_save_epoch_freq     = Config.NUM_SAVE_EPOCH_FREQ
    num_epoch_to_finish     = Config.NUM_EPOCH_TO_FINISH

    s_epoch         = 0
    step_accu       = 0
    iter_end        = sample_size * num_epoch_to_finish
    losses          = deque(maxlen=num_steps_to_display)
    time_checkpoint = time.time()

    if args.resume:
        s_epoch     = model.load(args.checkpoint_dir, optimizer, scheduler)
        #s_epoch     = model.load(args.checkpoint_dir, optimizer)
        step_accu   = sample_size * s_epoch
        pname       = args.checkpoint_dir + '/model-last.pt'
        Log.i(f'Model has been restored from file: {pname}')

    for epoch in range(s_epoch+1, num_epoch_to_finish+1):
        iter_batch = 0
        for _, (_, image_batch, _, bboxes_batch, labels_batch) in enumerate(dataloader):
            #batch_size      = image_batch.shape[0]
            image_batch     = image_batch.to(device)
            bboxes_batch    = bboxes_batch.to(device)
            labels_batch    = labels_batch.to(device)
            iter_batch     += Config.BATCH_SIZE
            step_accu      += Config.BATCH_SIZE

            '''###test
            gt_img  = visdom_bbox(image_batch, bboxes_batch[0], labels_batch[0])
            pname   = '{}/train_gt{}.png'.format(args.checkpoint_dir, str(step_accu))
            save_MNIST(gt_img, pname)
            '''

            anchor_cls_score_losses, \
            anchor_boxpred_losses, \
            proposal_class_losses, \
            proposal_boxpred_losses = model.train().forward(image_batch, bboxes_batch, labels_batch)

            anchor_cls_score_lossem = anchor_cls_score_losses.mean()
            anchor_boxpred_lossem   = anchor_boxpred_losses.mean()
            proposal_class_lossm    = proposal_class_losses.mean()
            proposal_boxpred_lossem = proposal_boxpred_losses.mean()

            loss = anchor_cls_score_lossem + anchor_boxpred_lossem + proposal_class_lossm + proposal_boxpred_lossem

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

            if iter_batch % num_steps_to_display == 0:
                elapsed_time    = time.time() - time_checkpoint
                time_checkpoint = time.time()
                steps_per_sec   = num_steps_to_display / elapsed_time
                samples_per_sec = Config.BATCH_SIZE * steps_per_sec
                remain_hours    = (iter_end - step_accu) / steps_per_sec / 3600
                lrate           = optimizer.param_groups[0]['lr']
                avg_loss        = sum(losses) / len(losses)
                Log.i(f'E/I:{epoch}/{iter_batch}, L.rate:{lrate:.6f}, Loss:{avg_loss:.6f}, {samples_per_sec:.2f} samples/sec, R.hours:{remain_hours:.1f}')

        if epoch % num_save_epoch_freq == 0:
            pname = model.save(args.checkpoint_dir, optimizer, scheduler, epoch)
            #pname = model.save(args.checkpoint_dir, optimizer, epoch=epoch)
            Log.i(f'Model has been saved to {pname}')

if __name__ == '__main__':
    #prefix = '{}'.format(time.strftime('%Y%m%d%H%M%S'))
    #path_to_checkpoints_dir = os.path.join(args.checkpoint_dir, prefix)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    Config.setup(image_min_side=args.image_min_side,
                 image_max_side=args.image_max_side,
                 anchor_ratios=args.anchor_ratios,
                 anchor_sizes=args.anchor_sizes,
                 #pooler_mode=args.pooler_mode,
                 rpn_pre_nms_top_n=args.rpn_pre_nms_top_n,
                 rpn_post_nms_top_n=args.rpn_post_nms_top_n,
                 anchor_smooth_l1_loss_beta=args.anchor_smooth_l1_loss_beta,
                 proposal_smooth_l1_loss_beta=args.proposal_smooth_l1_loss_beta,
                 batch_size=args.batch_size,
                 learning_rate=args.learning_rate,
                 momentum=args.momentum,
                 weight_decay=args.weight_decay,
                 #step_lr_sizes=args.step_lr_sizes,
                 update_lr_freq=args.update_lr_freq,
                 step_lr_gamma=args.step_lr_gamma,
                 #warm_up_factor=args.warm_up_factor,
                 #warm_up_num_iters=args.warm_up_num_iters,
                 num_steps_to_display=args.num_steps_to_display,
                 num_save_epoch_freq=args.num_save_epoch_freq,
				 num_epoch_to_finish=args.num_epoch_to_finish)

    #Log.initialize(os.path.join(args.checkpoint_dir,'train.log'))
    prefix  = '{}'.format(time.strftime('%Y%m%d%H%M%S'))
    pname   = '{}/train-{}.log'.format(args.checkpoint_dir, prefix,)
    Log.initialize(pname)
    Log.i('Arguments:')
    for k, v in vars(args).items():
        Log.i(f'\t{k} = {v}')
    Log.i(Config.describe())

    _train()

    Log.i('Done')


