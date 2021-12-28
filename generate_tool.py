import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as tnf
from torchvision import ops
from typing import Tuple, List
from bbox import BBox

# refer to: simple-faster-rcnn/rpn/creator_tools.py
class GenerateTool(object):
    def __init__(self,
                 num_classes:       int,
                 anchor_ratios:     List,
                 anchor_sizes:      List,
                 pre_nms_top_n:     int,
                 post_nms_top_n:    int):

        self.num_classes                = num_classes
        #self._anchor_ratios             = np.array(anchor_ratios, dtype=np.float64)
        #self._anchor_sizes              = np.array(anchor_sizes,  dtype=np.float64) #16 * scale[8, 16, 32]
        self._anchor_ratios             = torch.from_numpy(np.array(anchor_ratios)).float()
        self._anchor_sizes              = torch.from_numpy(np.array(anchor_sizes)).float()
        self._pre_nms_top_n             = pre_nms_top_n
        self._post_nms_top_n            = post_nms_top_n
        self._detectbox_normalize_mean  = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float)
        self._detectbox_normalize_std   = torch.tensor([0.1, 0.1, 0.2, 0.2], dtype=torch.float)

    def anchors(self, image_width: int, image_height: int, num_x_anchors: int, num_y_anchors: int) -> Tensor:
        '''array[num_y_anchors + 2][1:-1] = array[1] ~ array[num_y_anchors+2-2]'''
        #y_center    = np.linspace(start=0, stop=image_height, num=num_y_anchors + 2)[1:-1]
        #x_center    = np.linspace(start=0, stop=image_width,  num=num_x_anchors + 2)[1:-1]
        y_center    = torch.linspace(start=0, end=image_height, steps=num_y_anchors + 2)[1:-1]
        x_center    = torch.linspace(start=0, end=image_width, steps=num_x_anchors + 2)[1:-1]

        #ratios      = np.array(self._anchor_ratios)
        #ratios      = ratios[:, 0] / ratios[:, 1]
        #sizes       = np.array(self._anchor_sizes)

        '''np.set_printoptions(threshold=np.inf)
        print(x_center)
        print('')
        print(y_center)
        y_center , x_center = np.meshgrid(y_center, x_center, indexing='ij')
        print(x_center)
        print('')
        print(y_center)
        plt.plot(x_center, y_center, marker='.', color='r', linestyle='none')
        plt.show()'''

        # combine x[], y[] to a mesh grid
        #y_out, x_out, ratios, sizes = np.meshgrid(y_center, x_center, self._anchor_ratios, self._anchor_sizes, indexing='ij')
        y_out, x_out, ratios, sizes = torch.meshgrid(y_center, x_center, self._anchor_ratios, self._anchor_sizes)

        # to 1d
        y_out   = y_out.reshape(-1)
        x_out   = x_out.reshape(-1)
        ratios  = ratios.reshape(-1)
        sizes   = sizes.reshape(-1)

        #widths  = sizes * np.sqrt(1.0 / ratios)
        #heights = sizes * np.sqrt(ratios)
        h_ratios = torch.sqrt(ratios) #faster way
        heights  = sizes * h_ratios
        widths   = sizes * (1.0 / h_ratios)

        #center_based_anchor_bboxes  = np.stack((x_out, y_out, widths, heights), axis=1)
        #center_based_anchor_bboxes  = torch.from_numpy(center_based_anchor_bboxes).float()
        center_based_anchor_bboxes  = torch.stack((x_out, y_out, widths, heights), dim=1)
        anchor_gen_bboxes           = BBox.center_to_ltrb(center_based_anchor_bboxes)

        return anchor_gen_bboxes

    def proposals(self,
                  anchor_gen_bboxes:    Tensor,
                  anchor_cls_score:     Tensor,
                  anchor_bboxdelta:     Tensor,
                  image_width:          int,
                  image_height:         int) -> Tensor:
        nms_proposal_bboxes_batch   = []
        padded_proposal_bboxes      = []

        batch_size          = anchor_gen_bboxes.shape[0]
        predltrb            = BBox.bboxdelta_to_predltrb(anchor_gen_bboxes, anchor_bboxdelta)
        proposal_bboxes     = BBox.clip(predltrb, left=0, top=0, right=image_width, bottom=image_height)
        proposal_fg_probs   = tnf.softmax(anchor_cls_score[:, :, 1], dim=-1)
        _, sorted_indices   = torch.sort(proposal_fg_probs, dim=-1, descending=True)

        threshold = 0.7
        for batch_index in range(batch_size):
            sorted_bboxes   = proposal_bboxes[batch_index][sorted_indices[batch_index]][:self._pre_nms_top_n]
            sorted_probs    = proposal_fg_probs[batch_index][sorted_indices[batch_index]][:self._pre_nms_top_n]
            kept_indices    = ops.nms(sorted_bboxes, sorted_probs, threshold)
            nms_bboxes      = sorted_bboxes[kept_indices][:self._post_nms_top_n] #keep the most is 2000 bboxes
            nms_proposal_bboxes_batch.append(nms_bboxes)

        #compare each list which have max len
        max_nms_proposal_bboxes_length = max([len(it) for it in nms_proposal_bboxes_batch])

        # if nms_proposal_bboxes not enough to max len then add zeros
        for nms_proposal_bboxes in nms_proposal_bboxes_batch:
            '''padded_proposal_bboxes.append(torch.cat([nms_proposal_bboxes,
                                                        torch.zeros(max_nms_proposal_bboxes_length - len(nms_proposal_bboxes), 4).to(nms_proposal_bboxes)]))'''
            remain = max_nms_proposal_bboxes_length - len(nms_proposal_bboxes)
            padded_proposal_bboxes.append(torch.cat([nms_proposal_bboxes, torch.zeros(remain,4).to(nms_proposal_bboxes)]))

        padded_proposal_bboxes = torch.stack(padded_proposal_bboxes, dim=0)

        return padded_proposal_bboxes

    def detections(self,
                   proposal_gen_bboxes:Tensor,
                   proposal_classes:   Tensor,
                   proposal_boxpred:   Tensor,
                   image_width:        int,
                   image_height:       int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size                  = proposal_gen_bboxes.shape[0]
        proposal_boxpred            = proposal_boxpred.view(batch_size, -1, self.num_classes, 4)
        detectbox_normalize_std     = self._detectbox_normalize_std.to(device=proposal_boxpred.device)
        detectbox_normalize_mean    = self._detectbox_normalize_mean.to(device=proposal_boxpred.device)
        proposal_boxpred            = proposal_boxpred * detectbox_normalize_std + detectbox_normalize_mean

        proposal_gen_bboxes         = proposal_gen_bboxes.unsqueeze(dim=2).repeat(1, 1, self.num_classes, 1)
        predltrb                    = BBox.bboxdelta_to_predltrb(proposal_gen_bboxes, proposal_boxpred)
        detection_bboxes            = BBox.clip(predltrb, left=0, top=0, right=image_width, bottom=image_height)
        detection_probs             = tnf.softmax(proposal_classes, dim=-1)

        all_detection_bboxes        = []
        all_detection_classes       = []
        all_detection_probs         = []
        all_detection_batch_indices = []

        threshold = 0.3
        for batch_index in range(batch_size):
            for c in range(1, self.num_classes):
                class_bboxes = detection_bboxes[batch_index, :, c, :]
                class_probs = detection_probs[batch_index, :, c]
                # kept_indices = nms(class_bboxes, class_probs, threshold)
                kept_indices    = ops.nms(class_bboxes, class_probs, threshold)
                class_bboxes    = class_bboxes[kept_indices]
                class_probs     = class_probs[kept_indices]

                all_detection_bboxes.append(class_bboxes)
                all_detection_classes.append(torch.full((len(kept_indices),), c, dtype=torch.int))
                all_detection_probs.append(class_probs)
                all_detection_batch_indices.append(torch.full((len(kept_indices),), batch_index, dtype=torch.long))

        all_detection_bboxes        = torch.cat(all_detection_bboxes, dim=0)
        all_detection_classes       = torch.cat(all_detection_classes, dim=0)
        all_detection_probs         = torch.cat(all_detection_probs, dim=0)
        all_detection_batch_indices = torch.cat(all_detection_batch_indices, dim=0)

        return all_detection_bboxes, all_detection_classes, all_detection_probs, all_detection_batch_indices