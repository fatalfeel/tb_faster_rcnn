import torch
from torch import nn, Tensor
from torch.nn import functional as tnf
from typing import Tuple, List, Optional, Union
from bbox import BBox
#from support.layer.nms import nms

class RegionProposalNetwork(nn.Module):
    def __init__(self,
                 num_features_out:          int,
                 anchor_ratios:             List,
                 anchor_sizes:              List,
                 anchor_smooth_l1_loss_beta:float):

        super().__init__()

        #self._anchor_ratios = anchor_ratios
        #self._anchor_sizes  = anchor_sizes
        num_anchor_ratios   = len(anchor_ratios)
        num_anchor_sizes    = len(anchor_sizes)
        num_anchors         = num_anchor_ratios * num_anchor_sizes

        self._anchor_smooth_l1_loss_beta = anchor_smooth_l1_loss_beta

        self._rpnconvseq        = nn.Sequential(nn.Conv2d(in_channels=num_features_out, out_channels=512, kernel_size=3, stride=1, padding=1),
                                                nn.ReLU())

        #each anchor is given a positive or negative objectness score based on the Intersection-over-Union (IoU).
        self._anchor_cls_score  = nn.Conv2d(in_channels=512, out_channels=num_anchors * 2, kernel_size=1, stride=1, padding=0) #*2 means positive or negative(bg)
        self._anchor_bboxdelta  = nn.Conv2d(in_channels=512, out_channels=num_anchors * 4, kernel_size=1, stride=1, padding=0) #*4 means dx,dy,dw,dh

    def forward(self,
                resnet_features:    Tensor,
                anchor_gen_bboxes:  Optional[Tensor]    = None,
                gt_bboxes_batch:    Optional[Tensor]    = None,
                image_width:        Optional[int]       = None,
                image_height:       Optional[int]       = None) -> Union[Tuple[Tensor, Tensor],
                                                                         Tuple[Tensor, Tensor, Tensor, Tensor]]:

        batch_size          = resnet_features.shape[0]
        rpn_features        = self._rpnconvseq(resnet_features) #features = self._rpnconvseq(resnet_output)

        '''https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html'''
        anchor_cls_score    = self._anchor_cls_score(rpn_features)
        anchor_bboxdelta    = self._anchor_bboxdelta(rpn_features)

        anchor_cls_score    = anchor_cls_score.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
        anchor_bboxdelta    = anchor_bboxdelta.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)

        # remove cross-boundary
        if self.training:
            #inside_indices = BBox.InsideBound(anchor_gen_bboxes, left=0, top=0, right=image_width, bottom=image_height).nonzero().unbind(dim=1)
            bb_insidebound  = BBox.InsideBound(anchor_gen_bboxes, left=0, top=0, right=image_width, bottom=image_height)
            inside_indices  = torch.nonzero(bb_insidebound).unbind(dim=1) #split false part and true part

            # true part of inside_indices then bboxes get coordinate
            inside_anchor_gen_bboxes    = anchor_gen_bboxes[inside_indices].view(batch_size, -1, anchor_gen_bboxes.shape[2])
            inside_anchor_cls_score     = anchor_cls_score[inside_indices].view(batch_size, -1, anchor_cls_score.shape[2])
            inside_anchor_bboxdelta     = anchor_bboxdelta[inside_indices].view(batch_size, -1, anchor_bboxdelta.shape[2])

            # torch.full = fill -1 of inside_anchor_gen_bboxes.size(1)
            labels  = torch.full((batch_size, inside_anchor_gen_bboxes.shape[1]),
                                 -1,
                                 dtype  = torch.long,
                                 device = inside_anchor_gen_bboxes.device)

            ious    = BBox.getIoUs(inside_anchor_gen_bboxes, gt_bboxes_batch) #gt = ground truth
            #anchor_max_ious, anchor_assignments = ious.max(dim=2)
            #gt_max_ious, gt_assignments         = ious.max(dim=1)
            gt_max_ious,    gt_assignments      = torch.max(ious, dim=1) #column max
            anchor_max_ious,anchor_assignments  = torch.max(ious, dim=2) #row max

            #anchor_additions = ((ious > 0) & (ious == gt_max_ious.unsqueeze(dim=1))).nonzero()[:, :2].unbind(dim=1)
            max_ious            = (ious > 0) & (ious == gt_max_ious.unsqueeze(dim=1))   #max value is true, others false
            anchor_additions    = torch.nonzero(max_ious)[:, :2].unbind(dim=1)          #get dim[0] dim[1] of nonzero array into tuple (dim[0],dim[1])
            labels[anchor_max_ious < 0.3] = 0
            labels[anchor_additions] = 1
            labels[anchor_max_ious >= 0.7] = 1

            # select 256 x `batch_size` samples
            '''
            fg_indices = (labels == 1).nonzero()
            bg_indices = (labels == 0).nonzero()
            '''
            fg_indices = torch.nonzero(labels > 0)
            bg_indices = torch.nonzero(labels == 0)
            ### test only
            '''fg_indices_777 = torch.nonzero(labels == 1)
            for k in range(fg_indices_777.shape[0]):
                if fg_indices_777[k][1] != fg_indices[k][1]:
                    print('fg_indices_xxx error')'''

            #fg_samples = fg_indices[torch.randperm(len(fg_indices))[:min(len(fg_indices), 128 * batch_size)]]
            #bg_samples = bg_indices[torch.randperm(len(bg_indices))[:256 * batch_size - len(fg_samples)]]
            fg_rand     = torch.randperm(len(fg_indices))           #random number 1~len
            fg_size     = min(len(fg_indices), 128 * batch_size)    #select min value
            fg_range    = fg_rand[:fg_size]                         #pick 0~fg_size of random number
            fg_samples  = fg_indices[fg_range]

            bg_rand     = torch.randperm(len(bg_indices))
            bg_size     = 256 * batch_size - len(fg_samples)
            bg_range    = bg_rand[:bg_size]
            bg_samples  = bg_indices[bg_range]

            #selected_indices = torch.cat([fg_samples, bg_samples], dim=0)
            #selected_indices = selected_indices[torch.randperm(len(selected_indices))].unbind(dim=1)
            fgbg_samples        = torch.cat([fg_samples, bg_samples], dim=0)
            selected_rand       = torch.randperm(len(fgbg_samples))
            selected_indices    = fgbg_samples[selected_rand].unbind(dim=1)

            inside_anchor_gen_bboxes    = inside_anchor_gen_bboxes[selected_indices]
            gt_bboxes                   = gt_bboxes_batch[selected_indices[0], anchor_assignments[selected_indices]]
            gt_anchor_labels            = labels[selected_indices]
            gt_anchor_offset            = BBox.offset_from_gt_center(inside_anchor_gen_bboxes, gt_bboxes)
            batch_indices               = selected_indices[0]

            anchor_cls_score_losses, anchor_bboxdelta_losses = self.getLoss(inside_anchor_cls_score[selected_indices],
                                                                            inside_anchor_bboxdelta[selected_indices],
                                                                            gt_anchor_labels,
                                                                            gt_anchor_offset,
                                                                            batch_size,
                                                                            batch_indices)

            return anchor_cls_score, anchor_bboxdelta, anchor_cls_score_losses, anchor_bboxdelta_losses

        else:
            return anchor_cls_score, anchor_bboxdelta

    def getLoss(self,
                anchor_cls_score: Tensor,
                anchor_bboxdelta: Tensor,
                gt_anchor_labels: Tensor,
                gt_anchor_offset: Tensor,
                batch_size: int,
                batch_indices: Tensor) -> Tuple[Tensor, Tensor]:

        cross_entropies     = torch.empty(batch_size,   dtype=torch.float, device=anchor_cls_score.device)
        smooth_l1_losses    = torch.empty(batch_size,  dtype=torch.float, device=anchor_bboxdelta.device)

        for batch_index in range(batch_size):
            #selected_indices = (batch_indices == batch_index).nonzero().view(-1)
            selected_indices    = torch.nonzero(batch_indices == batch_index).view(-1)

            cross_entropy       = tnf.cross_entropy(input  = anchor_cls_score[selected_indices],
                                                    target = gt_anchor_labels[selected_indices])

            #fg_indices = gt_anchor_labels[selected_indices].nonzero().view(-1)
            fg_indices = torch.nonzero(gt_anchor_labels[selected_indices]).view(-1)

            '''
            pred:
            dx(p) = (ĝx-px)/pw
            dy(p) = (ĝy-py)/ph
            dw(p) = ln(ĝw/pw)
            dh(p) = ln(ĝh/ph)
            offset:
            tx = (gx−px)/pw
            ty = (gy−py)/ph
            tw = ln(gw/pw)
            th = ln(gh/ph)
            '''
            smooth_l1_loss = self.beta_smooth_l1_loss(  pred    = anchor_bboxdelta[selected_indices][fg_indices],
                                                        offset  = gt_anchor_offset[selected_indices][fg_indices],
                                                        beta    = self._anchor_smooth_l1_loss_beta  )

            cross_entropies[batch_index]    = cross_entropy
            smooth_l1_losses[batch_index]   = smooth_l1_loss

        return cross_entropies, smooth_l1_losses

    def beta_smooth_l1_loss(self, pred: Tensor, offset: Tensor, beta: float) -> Tensor:
        diff = torch.abs(pred - offset)
        # loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
        # flag = diff < beta
        # (flag * 0.5 * (diff ** 2)/beta + (1 - flag) * (diff - 0.5 * beta)
        loss = torch.where(diff < beta, 0.5 * (diff ** 2) / beta, diff - 0.5 * beta)
        loss = loss.sum() / (pred.numel() + 1e-8)

        return loss
