from .densenet import densenet121, densenet161, densenet169
from .googlenet import googlenet
from .inception import inception_v3
from .mobilenetv2 import mobilenet_v2
from .resnet import resnet18, resnet34, resnet50
from .vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

all_classifiers = {
    "vgg11_bn": vgg11_bn,
    "vgg13_bn": vgg13_bn,
    "vgg16_bn": vgg16_bn,
    "vgg19_bn": vgg19_bn,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "densenet121": densenet121,
    "densenet161": densenet161,
    "densenet169": densenet169,
    "mobilenet_v2": mobilenet_v2,
    "googlenet": googlenet,
    "inception_v3": inception_v3,
}