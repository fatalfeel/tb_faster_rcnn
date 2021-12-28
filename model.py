import os
import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Tuple, List, Optional
from backbone.basenet import BackboneBase
from generate_tool import GenerateTool
from region_proposal_network import RegionProposalNetwork
from detection import Detection

class Model(nn.Module):
    def __init__(self,
                 backbone:                      BackboneBase,
                 num_classes:                   int,
                 #pooler_mode:                  Pooler.Mode,
                 anchor_ratios:                 List,
                 anchor_sizes:                  List,
                 rpn_pre_nms_top_n:             int,
                 rpn_post_nms_top_n:            int,
                 anchor_smooth_l1_loss_beta:    Optional[float] = None,
                 proposal_smooth_l1_loss_beta:  Optional[float] = None):
        super().__init__()

        self.resnet, hidden_layer, num_resnet_features_out, num_hidden_out = backbone.features()
        self._bn_modules = nn.ModuleList([it for it in self.resnet.modules() if isinstance(it, nn.BatchNorm2d)]
                                         +
                                         [it for it in hidden_layer.modules() if isinstance(it, nn.BatchNorm2d)])

        # NOTE: It's crucial to freeze batch normalization modules for few batches training, which can be done by following processes
        #       (1) Change mode to `eval`
        #       (2) Disable gradient (we move this process into `forward`)
        for bn_module in self._bn_modules:
            for parameter in bn_module.parameters():
                parameter.requires_grad = False

        #self.rpn        = RegionProposalNetwork(num_resnet_features_out, anchor_ratios, anchor_sizes, rpn_pre_nms_top_n, rpn_post_nms_top_n, anchor_smooth_l1_loss_beta)
        self.rpn        = RegionProposalNetwork(num_resnet_features_out, anchor_ratios, anchor_sizes, anchor_smooth_l1_loss_beta)
        self.detection  = Detection(hidden_layer, num_hidden_out, num_classes, proposal_smooth_l1_loss_beta)
        self.gtool      = GenerateTool(num_classes, anchor_ratios, anchor_sizes, rpn_pre_nms_top_n, rpn_post_nms_top_n)

    def forward(self,
                image_batch: Tensor,
                gt_bboxes_batch: Tensor = None,
                gt_labels_batch: Tensor = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        # disable gradient for each forwarding BatchNorm2d just in case model was switched to `train` mode at any time
        for bn_module in self._bn_modules:
            bn_module.eval()

        # resnet_features = resnet_output
        resnet_features = self.resnet(image_batch)

        batch_size, _, image_height, image_width            = image_batch.shape
        _, _, resnet_features_height, resnet_features_width = resnet_features.shape

        anchor_gen_bboxes = self.gtool.anchors(image_width,
                                               image_height,
                                               num_x_anchors=resnet_features_width,
                                               num_y_anchors=resnet_features_height ).to(resnet_features).repeat(batch_size, 1, 1)

        if self.training:
            anchor_cls_score, \
            anchor_bboxdelta, \
            anchor_cls_score_losses, \
            anchor_bboxdelta_losses = self.rpn.forward(resnet_features,
                                                       anchor_gen_bboxes,
                                                       gt_bboxes_batch,
                                                       image_width,
                                                       image_height)

            #it's necessary to detach `proposal_gen_bboxes` here
            proposal_gen_bboxes = self.gtool.proposals(anchor_gen_bboxes,
                                                       anchor_cls_score,
                                                       anchor_bboxdelta,
                                                       image_width,
                                                       image_height).detach()

            proposal_classes, \
            proposal_boxdelta, \
            proposal_class_losses, \
            proposal_boxdelta_losses = self.detection.forward(resnet_features,
                                                              proposal_gen_bboxes,
                                                              gt_bboxes_batch,
                                                              gt_labels_batch)

            return anchor_cls_score_losses, \
                   anchor_bboxdelta_losses, \
                   proposal_class_losses, \
                   proposal_boxdelta_losses
        else:
            anchor_cls_score, anchor_bboxdelta = self.rpn.forward(resnet_features)

            proposal_gen_bboxes = self.gtool.proposals(anchor_gen_bboxes,
                                                       anchor_cls_score,
                                                       anchor_bboxdelta,
                                                       image_width,
                                                       image_height)

            proposal_classes, proposal_boxdelta = self.detection.forward(resnet_features, proposal_gen_bboxes)

            detection_bboxes, \
            detection_classes, \
            detection_probs, \
            detection_batch_indices = self.gtool.detections(proposal_gen_bboxes,
                                                            proposal_classes,
                                                            proposal_boxdelta,
                                                            image_width,
                                                            image_height)

            return detection_bboxes, \
                   detection_classes, \
                   detection_probs, \
                   detection_batch_indices

    def save(self, checkpoint_dir: str, optimizer: Optimizer = None, scheduler: _LRScheduler = None, epoch: int = 0) -> str:
        if scheduler is None:
            checkpoint = {'state_dict':             self.state_dict(),
                          'optimizer_state_dict':   optimizer.state_dict(),
                          'epoch':                  epoch}
        else:
            checkpoint = {'state_dict':             self.state_dict(),
                          'optimizer_state_dict':   optimizer.state_dict(),
                          'scheduler_state_dict':   scheduler.state_dict(),
                          'epoch':                  epoch}

        pname       = os.path.join(checkpoint_dir, f'model-{epoch}.pt')
        torch.save(checkpoint, pname)

        lastname    = os.path.join(checkpoint_dir, 'model-last.pt')
        torch.save(checkpoint, lastname)

        return pname

    def load(self, checkpoint_dir: str, optimizer: Optimizer = None, scheduler: _LRScheduler = None) -> 'Model':
        lastname    = os.path.join(checkpoint_dir, 'model-last.pt')
        checkpoint  = torch.load(lastname)

        self.load_state_dict(checkpoint['state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        epoch = checkpoint['epoch']

        return epoch
