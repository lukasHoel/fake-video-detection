"""
Author: Anna Mittermair
"""


from networks.xception import xception
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch
import numpy as np


class ConcatBaselineModel(nn.Module):
    """
    Two stage model:
    Stage 1: Feature extraction on every frame using pretrained feature extraction networks, one per frame.
    Stage 2: Concatenate features along a new dimension, downsample channels, then use several CNN layers.
    Stage 3: Classification (FC) layer
    
    
    Feature extraction blocks taken from networks/baseline.py 
        --> (Adapted slightly from FaceForensics version: https://github.com/ondyari/FaceForensics/blob/master/classification/network/models.py)
    """

    def __init__(self, model_choice='xception', num_frames=5, drop_blocks=8, image_shape=299):
        super(ConcatBaselineModel, self).__init__()
        self.num_frames = num_frames
        self.model_choice = model_choice
        self.first_stage_models = []
        first_stage_out_ftrs = 0
        
        # Feature extraction networks, one per frame
        for i in range(num_frames):
            if model_choice == 'xception':
                model = xception()

            elif model_choice == 'resnet50' or model_choice == 'resnet18':
                if model_choice == 'resnet50':
                    model = torchvision.models.resnet50(pretrained=True)
                if model_choice == 'resnet18':
                    model = torchvision.models.resnet18(pretrained=True)
            else:
                raise Exception('Choose valid model, e.g. resnet50')


            model = torch.nn.Sequential(*(list(model.children())[:-drop_blocks]))
            test = torch.Tensor(np.ones((1, 3, image_shape, image_shape)))
            output_shape = model(test).shape
            output_image_size = output_shape[2]
            first_stage_out_ftrs = output_shape[1]

        
            self.first_stage_models.append(model)
        num_3d_layers = self.num_frames // 2
        self.temporal_encoder_input_channels = first_stage_out_ftrs
        self.num_channels = 64
        self.temporal_encoder_output_channels = (output_image_size - 3 * 2) ** 2 * self.num_channels

        self.first_stage = nn.ModuleList(self.first_stage_models)
        
        # Take only num_channels channels
        self.conv_red = nn.Conv3d(self.temporal_encoder_input_channels, self.num_channels, 1)
        self.conv_red_bn = nn.BatchNorm3d(self.num_channels)
        
        # CNN part on all frame features, 3D CNN because of new dimension for all frames
        self.conv1 = nn.Conv3d(self.num_channels, self.num_channels, 3)
        self.conv1_bn = nn.BatchNorm3d(self.num_channels)

        if num_3d_layers > 1:
            self.conv2 = nn.Conv3d(self.num_channels, self.num_channels, 3)
            self.conv2_bn = nn.BatchNorm3d(self.num_channels)
        else:
            self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, 3)
            self.conv2_bn = nn.BatchNorm2d(self.num_channels)
        if num_3d_layers > 2:
            self.conv3 = nn.Conv3d(self.num_channels, self.num_channels, 3)
            self.conv3_bn = nn.BatchNorm3d(self.num_channels)
        else:
            self.conv3 = nn.Conv2d(self.num_channels, self.num_channels, 3)
            self.conv3_bn = nn.BatchNorm2d(self.num_channels)
        
        # FC layer    
        self.fc_end = nn.Sequential(nn.Linear(self.temporal_encoder_output_channels, 2), nn.Softmax())


    def train_only_last_layer(self):
        self.set_trainable_up_to()

    def set_trainable_up_to(self):
        """
        Freezes all layers below a specific layer and sets the following layers
        to true if boolean else only the fully connected final layer
        :param train_rest: whether or not to train the following layers after layername
        :param layername: depends on network, for inception e.g. Conv2d_4a_3x3. If None: everything trainable
        :return:
        """
        # Stage-1: freeze all the layers
        for frame in range(self.num_frames):
            for _, param in self.first_stage_models[frame].named_parameters():
                param.requires_grad = False
            for _, param in self.fc_end.named_parameters():
                param.requires_grad = True

    def forward(self, x):
        y = []
        
        # Take video frames as input
        x = x["image"]
        
        # Extract features
        for i in range(self.num_frames):
            x_i = x[:, :, i, :, :]
            x_i = x_i.squeeze(dim=1)
            model_i = self.first_stage[i]
            y_i = model_i(x_i)
            image_size = y_i.shape[3]
            y.append(y_i.view((-1, self.temporal_encoder_input_channels, image_size, image_size, 1)))
            
        # Concatenate and reduce channels
        y = torch.cat(y, dim=4)
        y = F.relu(self.conv_red_bn(self.conv_red(y)))
        
        # CNN block 1
        if (y.shape[-1] == 1):
            y = y.view(-1, y.shape[1], y.shape[2], y.shape[3])
        y = F.relu(self.conv1_bn(self.conv1(y)))
        # CNN block 2
        y = y.view(-1, self.num_channels, image_size - 2, image_size - 2, self.num_frames - 2)
        if (y.shape[-1] == 1):
            y = y.view(-1, y.shape[1], y.shape[2], y.shape[3])
        y = F.relu(self.conv2_bn(self.conv2(y)))
        # CNN block 3
        if (y.shape[-1] == 1):
            y = y.view(-1, y.shape[1], y.shape[2], y.shape[3])
        y = F.relu(self.conv3_bn(self.conv3(y)))
        
        # Reshape and apply fc layer
        if (y.shape[-1] == 1):
            y = y.view(-1, y.shape[1], y.shape[2], y.shape[3])
        y = y.flatten(start_dim=1)
        y = self.fc_end(y)
        return y


if __name__ == '__main__':
    model = ConcatBaselineModel()
    model.train_only_last_layer()
    print(model)