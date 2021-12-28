import torch
from torch import Tensor
from typing import List

class BBox(object):
    def __init__(self, left: float, top: float, right: float, bottom: float):
        super().__init__()
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def __repr__(self) -> str:
        return 'BBox[l={:.1f}, t={:.1f}, r={:.1f}, b={:.1f}]'.format(
            self.left, self.top, self.right, self.bottom)

    def tolist(self) -> List[float]:
        return [self.left, self.top, self.right, self.bottom]

    '''(x0,y0,x1,y1) to (c0,c1,w,h)'''
    @staticmethod
    #def to_center_base(bboxes: Tensor):
    def ltrb_to_center(bboxes: Tensor) -> Tensor:
        center_wh = torch.stack([(bboxes[..., 0] + bboxes[..., 2]) / 2,
                                 (bboxes[..., 1] + bboxes[..., 3]) / 2,
                                 bboxes[..., 2] - bboxes[..., 0],
                                 bboxes[..., 3] - bboxes[..., 1]],
                                 dim=-1)

        return center_wh

    '''(c0,c1,w,h) to (x0,y0,x1,y1)'''
    @staticmethod
    #def from_center_base(center_based_bboxes: Tensor) -> Tensor:
    def center_to_ltrb(center_based_bboxes: Tensor) -> Tensor:
        '''return torch.stack([center_based_bboxes[..., 0] - center_based_bboxes[..., 2] / 2,
                            center_based_bboxes[..., 1] - center_based_bboxes[..., 3] / 2,
                            center_based_bboxes[..., 0] + center_based_bboxes[..., 2] / 2,
                            center_based_bboxes[..., 1] + center_based_bboxes[..., 3] / 2],
                            dim=-1)'''
        half_width  = center_based_bboxes[..., 2] / 2
        half_height = center_based_bboxes[..., 3] / 2
        return torch.stack([center_based_bboxes[..., 0] - half_width,
                            center_based_bboxes[..., 1] - half_height,
                            center_based_bboxes[..., 0] + half_width,
                            center_based_bboxes[..., 1] + half_height],
                            dim=-1)

    '''
    https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html
    https://blog.csdn.net/qq_34106574/article/details/81669891
    proposal to ground truth shift and scale
    tx = (gx−px)/pw
    ty = (gy−py)/ph
    tw = ln(gw/pw)
    th = ln(gh/ph)
    '''
    @staticmethod
    #def calc_transformer(gen_bboxes: Tensor, gt_bboxes: Tensor) -> Tensor:
    def offset_from_gt_center(gen_bboxes: Tensor, gt_bboxes: Tensor) -> Tensor:
        gen_bboxes_centerbase   = BBox.ltrb_to_center(gen_bboxes)
        gt_bboxes_centerbase    = BBox.ltrb_to_center(gt_bboxes)
        # torch.log is ln
        gt_offset = torch.stack([(gt_bboxes_centerbase[..., 0] - gen_bboxes_centerbase[..., 0]) / gen_bboxes_centerbase[..., 2],
                                 (gt_bboxes_centerbase[..., 1] - gen_bboxes_centerbase[..., 1]) / gen_bboxes_centerbase[..., 3],
                                 torch.log(gt_bboxes_centerbase[..., 2] / gen_bboxes_centerbase[..., 2]),
                                 torch.log(gt_bboxes_centerbase[..., 3] / gen_bboxes_centerbase[..., 3])],
                                 dim=-1)
        return gt_offset

    '''
    https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html
    https://blog.csdn.net/qq_34106574/article/details/81669891
    proposal and bboxdelta to predict ground truth
    ĝx = dx(p) * pw + px , dx(p) = (ĝx-px)/pw = bboxdelta[..., 0]
    ĝy = dy(p) * ph + py , dy(p) = (ĝy−py)/ph = bboxdelta[..., 1]
    ĝw = exp^(dw(p)) * pw, dw(p) = ln(ĝw/pw)  = bboxdelta[..., 2]
    ĝh = exp^(dh(p)) * ph, dh(p) = ln(ĝh/ph)  = bboxdelta[..., 3]
    '''
    @staticmethod
    #def apply_transformer(src_bboxes: Tensor, transformers: Tensor) -> Tensor:
    def bboxdelta_to_predltrb(gen_bboxes: Tensor, bboxdelta: Tensor) -> Tensor:
        gen_bboxes_centerbase = BBox.ltrb_to_center(gen_bboxes)
        # torch.exp is e^n,
        predcenter = torch.stack([bboxdelta[..., 0] * gen_bboxes_centerbase[..., 2] + gen_bboxes_centerbase[..., 0],
                                  bboxdelta[..., 1] * gen_bboxes_centerbase[..., 3] + gen_bboxes_centerbase[..., 1],
                                  torch.exp(bboxdelta[..., 2]) * gen_bboxes_centerbase[..., 2],
                                  torch.exp(bboxdelta[..., 3]) * gen_bboxes_centerbase[..., 3]],
                                  dim=-1)

        predltrb = BBox.center_to_ltrb(predcenter)

        return predltrb

    @staticmethod
    def getIoUs(source: Tensor, gtboxes: Tensor) -> Tensor:
        #source, gtboxes = source.unsqueeze(dim=-2).repeat(1, 1, gtboxes.shape[-2], 1), gtboxes.unsqueeze(dim=-3).repeat(1, source.shape[-2], 1, 1)
        source_ext      = source.unsqueeze(dim=-2).repeat(1, 1, gtboxes.shape[-2], 1)
        gtboxes_ext     = gtboxes.unsqueeze(dim=-3).repeat(1, source.shape[-2], 1, 1)

        #get max point
        intersection_left   = torch.max(source_ext[..., 0], gtboxes_ext[..., 0])
        intersection_top    = torch.max(source_ext[..., 1], gtboxes_ext[..., 1])
        intersection_right  = torch.min(source_ext[..., 2], gtboxes_ext[..., 2])
        intersection_bottom = torch.min(source_ext[..., 3], gtboxes_ext[..., 3])
        intersection_width  = torch.clamp(intersection_right - intersection_left, min=0)
        intersection_height = torch.clamp(intersection_bottom - intersection_top, min=0)
        intersection_area   = intersection_width * intersection_height

        # source and gtboxes rectangle area
        source_area     = (source_ext[..., 2] - source_ext[..., 0]) * (source_ext[..., 3] - source_ext[..., 1])
        gtboxes_area    = (gtboxes_ext[..., 2] - gtboxes_ext[..., 0]) * (gtboxes_ext[..., 3] - gtboxes_ext[..., 1])
        ious            = intersection_area / (source_area + gtboxes_area - intersection_area)

        return ious

    @staticmethod
    def InsideBound(bboxes: Tensor, left: float, top: float, right: float, bottom: float) -> Tensor:
        #return ((bboxes[..., 0] >= left) * (bboxes[..., 1] >= top) * (bboxes[..., 2] <= right) * (bboxes[..., 3] <= bottom))
        b0  = bboxes[..., 0] >= left
        b1  = bboxes[..., 1] >= top
        b2  = bboxes[..., 2] <= right
        b3  = bboxes[..., 3] <= bottom
        ret = b0*b1*b2*b3

        return ret

    @staticmethod
    def clip(bboxes: Tensor, left: float, top: float, right: float, bottom: float) -> Tensor:
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]].clamp(min=left, max=right)
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]].clamp(min=top, max=bottom)

        return bboxes
