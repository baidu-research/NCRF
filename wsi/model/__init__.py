import numpy as np

np.random.seed(0)

from wsi.model.resnet import (resnet18, resnet34, resnet50, resnet101,
                              resnet152)


MODELS = {'resnet18': resnet18,
          'resnet34': resnet34,
          'resnet50': resnet50,
          'resnet101': resnet101,
          'resnet152': resnet152}
