import torchvision
from torch import nn
from typing import Tuple

class ResNext101_32x8d(object):
    def __init__(self, pretrained: bool):
        super().__init__()
        self._pretrained = pretrained

    def features(self) -> Tuple[nn.Module, nn.Module, int, int]:
        resnext101_32x8d = torchvision.models.resnext101_32x8d(pretrained=self._pretrained)

        # list(resnet101.children()) consists of following modules
        #   [0] = Conv2d, [1] = BatchNorm2d, [2] = ReLU,
        #   [3] = MaxPool2d, [4] = Sequential(Bottleneck...),
        #   [5] = Sequential(Bottleneck...),
        #   [6] = Sequential(Bottleneck...),
        #   [7] = Sequential(Bottleneck...),
        #   [8] = AvgPool2d, [9] = Linear
        children = list(resnext101_32x8d.children())
        features = children[:-3]
        num_features_out = 1024

        hidden = children[-3]
        num_hidden_out = 2048

        # from children_00 ~ children_04
        '''for parameters in [feature.parameters() for i, feature in enumerate(features) if i <= 4]:
            for parameter in parameters:
                parameter.requires_grad = False'''

        for i, feature in enumerate(features):
            if i <= 3:
                for parameters in [feature.parameters()]:
                    for parameter in parameters:
                        parameter.requires_grad = False
            else:
                break

        features = nn.Sequential(*features)

        return features, hidden, num_features_out, num_hidden_out
