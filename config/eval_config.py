from typing import List
from config.config import Config

class EvalConfig(Config):
    RPN_PRE_NMS_TOP_N: int  = 6000 #smaller can do faster
    #RPN_POST_NMS_TOP_N: int = 300
    RPN_POST_NMS_TOP_N: int = 1000 #smaller can do faster
    BATCH_SIZE: int         = 1

    @classmethod
    def setup(cls,
              image_min_side: float    = None,
              image_max_side: float    = None,
              anchor_ratios: List      = None,
              anchor_sizes: List       = None,
              #pooler_mode: str = None,
              rpn_pre_nms_top_n: int   = None,
              rpn_post_nms_top_n: int  = None):
        #super().setup(image_min_side, image_max_side, anchor_ratios, anchor_sizes, pooler_mode)
        super().setup(image_min_side, image_max_side, anchor_ratios, anchor_sizes)

        if rpn_pre_nms_top_n is not None:
            cls.RPN_PRE_NMS_TOP_N = rpn_pre_nms_top_n
        if rpn_post_nms_top_n is not None:
            cls.RPN_POST_NMS_TOP_N = rpn_post_nms_top_n
