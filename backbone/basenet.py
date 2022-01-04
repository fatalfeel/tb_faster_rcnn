from typing import Type
from backbone.resnet18 import ResNet18
from backbone.resnet50 import ResNet50
from backbone.resnet101 import ResNet101
from backbone.resnext101_32x8d import ResNext101_32x8d

class BackboneBase:
    OPTIONS = ['resnet18', 'resnet50', 'resnet101', 'resnext101_32x8d']
    @staticmethod
    def from_name(name: str) -> Type['Base']:
        if name == 'resnet18':
            return ResNet18
        elif name == 'resnet50':
            return ResNet50
        elif name == 'resnet101':
            return ResNet101
        elif name == 'resnext101_32x8d':
            return ResNext101_32x8d
        else:
            raise ValueError

    '''def __init__(self):
        super().__init__()

    def features(self) -> Tuple[nn.Module, nn.Module, int, int]:
        raise NotImplementedError'''