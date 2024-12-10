import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

from phase_classification import phase_classification_settings


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    else:
        device = torch.device("cpu")
    return device


class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=5, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output


class SimpleNet(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleNet, self).__init__()

        # Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=1, out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)
        self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.unit4 = Unit(in_channels=32, out_channels=64)
        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)
        self.unit7 = Unit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.unit8 = Unit(in_channels=64, out_channels=128)
        self.unit9 = Unit(in_channels=128, out_channels=128)
        self.unit10 = Unit(in_channels=128, out_channels=128)
        self.unit11 = Unit(in_channels=128, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.unit12 = Unit(in_channels=128, out_channels=128)
        self.unit13 = Unit(in_channels=128, out_channels=128)
        self.unit14 = Unit(in_channels=128, out_channels=128)

        self.avgpool = nn.AvgPool2d(kernel_size=4)

        # Add all the units into the Sequential layer in exact order
        # self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6
        #                          , self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
        #                          self.unit12, self.unit13, self.unit14, self.avgpool)
        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6
                                 , self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11,
                                 self.avgpool)

        self.fc1 = nn.Linear(in_features=128 * 12 * 12, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        # print("Output shape of Convolutional layers: {}".format(output.shape))
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output


def init_model(log=True):
    _model = ResNet18(num_classes=phase_classification_settings.num_classes, feature_extract=False,
                      use_pretrained=False)

    if torch.cuda.is_available():
        _model.to(get_device())
    if log:
        summary(_model,
                (3, phase_classification_settings.model_input_size, phase_classification_settings.model_input_size))
        # print(_model)
    return _model


def load_model_state_dict(_model, model_state_dict_path=phase_classification_settings.best_phase_model_path):
    _model.load_state_dict(torch.load(model_state_dict_path))
    return _model


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def ResNet18(num_classes, feature_extract, use_pretrained=False):
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    # for name, child in model_ft.named_children():
    #     if name in ['layer4']:
    #         print(name + ' has been unfrozen.')
    #         for param in child.parameters():
    #             param.requires_grad = True
    #     else:
    #         for param in child.parameters():
    #             param.requires_grad = False

    return model_ft


if __name__ == "__main__":
    # model = SimpleNet(num_classes=4)
    # summary(model, (1, 256, 256))

    model = ResNet18(num_classes=4, feature_extract=False, use_pretrained=False)
    summary(model, (3, phase_classification_settings.model_input_size, phase_classification_settings.model_input_size))
    # print(model)
    print("Done!")
