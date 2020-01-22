"""
Author: Lukas Hoellein

Adapted slightly from FaceForensics version: https://github.com/ondyari/FaceForensics/blob/master/classification/network/models.py
"""

from networks.xception import xception
import torch.nn as nn
import torchvision

class BaselineModel(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes

    Adapted slightly from FaceForensics version: https://github.com/ondyari/FaceForensics/blob/master/classification/network/models.py
    """
    def __init__(self, model_choice='xception', num_out_classes=2, dropout=0.0):
        super(BaselineModel, self).__init__()
        self.model_choice = model_choice

        if model_choice == 'xception':
            self.model = xception()
            # Replace fc
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
            else:
                print('Using dropout', dropout)
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )

        elif model_choice == 'resnet50' or model_choice == 'resnet18':
            if model_choice == 'resnet50':
                self.model = torchvision.models.resnet50(pretrained=True)
            if model_choice == 'resnet18':
                self.model = torchvision.models.resnet18(pretrained=True)
            # Replace fc
            num_ftrs = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_ftrs, num_out_classes)
            else:
                self.model.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        else:
            raise Exception('Choose valid model, e.g. resnet50')

    def train_only_last_layer(self):
        self.set_trainable_up_to(train_rest=False, layername="dummy_not_none")

    def set_trainable_up_to(self, train_rest=False, layername="last_linear"):
        """
        Freezes all layers below a specific layer and sets the following layers
        to true if boolean else only the fully connected final layer
        :param train_rest: whether or not to train the following layers after layername
        :param layername: depends on network, for inception e.g. Conv2d_4a_3x3. If None: everything trainable
        :return:
        """
        # Stage-1: freeze all the layers
        if layername is None:
            for i, param in self.model.named_parameters():
                param.requires_grad = True
                return
        else:
            for i, param in self.model.named_parameters():
                param.requires_grad = False
        if train_rest:
            # Make all layers following the layername layer trainable
            ct = []
            found = False
            for name, child in self.model.named_children():
                if layername in ct:
                    found = True
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)
            if not found:
                raise Exception('Layer not found, cant finetune!'.format(
                    layername))
        else:
            if self.model_choice == 'xception':
                # Make fc trainable
                for param in self.model.last_linear.parameters():
                    param.requires_grad = True

            else:
                # Make fc trainable
                for param in self.model.fc.parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    baseline = BaselineModel(model_choice='xception', num_out_classes=2, dropout=0.0)
    baseline.train_only_last_layer()
    print(baseline)

    for name, param in baseline.model.named_parameters():
        if param.requires_grad:
            print("param: {} requires_grad: {}".format(name, param.requires_grad))