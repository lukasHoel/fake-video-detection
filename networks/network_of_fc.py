from networks.xception import xception
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch
import numpy as np


class OFModel(nn.Module):
    """
       """

    def __init__(self, model_choice='xception', num_frames=5, drop_blocks=8, image_shape=299):
        super(OFModel, self).__init__()
        self.num_frames = num_frames
        self.model_choice = model_choice
        self.feature_extraction_frames = []
        self.feature_extraction_flow = []
        self.conv_block_frames = []
        self.conv_block_flow = []
        num_3d_layers = self.num_frames // 2
        first_stage_out_ftrs = 0

        # Feature extraction parts for frames
        for i in range(num_frames):
            model = self.make_feature_extraction_model(drop_blocks, model_choice)
            test = torch.Tensor(np.ones((1, 3, image_shape, image_shape)))
            output_shape = model(test).shape
            output_image_size = output_shape[2]
            first_stage_out_ftrs = output_shape[1]
            self.feature_extraction_frames.append(model)

        self.feature_extraction_frames = nn.ModuleList(self.feature_extraction_frames)

        self.temporal_encoder_input_channels = first_stage_out_ftrs
        self.num_channels = 64
        self.temporal_encoder_output_channels = (output_image_size - 3 * 2) ** 2 * self.num_channels

        self.conv_block_frames = self.make_conv_block(num_3d_layers)
        # OF
        for i in range(num_frames):
            model = self.make_feature_extraction_model(drop_blocks, model_choice)

            self.feature_extraction_flow.append(model)
        self.feature_extraction_flow = nn.ModuleList(self.feature_extraction_flow)
        self.conv_block_flow = self.make_conv_block(num_3d_layers)

        #Final fc layer
        self.fc_end = nn.Sequential(nn.Linear(self.temporal_encoder_output_channels * 2, 2), nn.Softmax())

    def make_conv_block(self, num_3d_layers):
        # Channel reduction layer
        conv_red = nn.Conv3d(self.temporal_encoder_input_channels, self.num_channels, 1)
        conv_red_bn = nn.BatchNorm3d(self.num_channels)

        # Conv block 1
        conv1 = nn.Conv3d(self.num_channels, self.num_channels, 3)
        conv1_bn = nn.BatchNorm3d(self.num_channels)

        # Conv block 2
        if num_3d_layers > 1:
            conv2 = nn.Conv3d(self.num_channels, self.num_channels, 3)
            conv2_bn = nn.BatchNorm3d(self.num_channels)
        else:
            conv2 = nn.Conv2d(self.num_channels, self.num_channels, 3)
            conv2_bn = nn.BatchNorm2d(self.num_channels)

        # Conv block 3
        if num_3d_layers > 2:
            conv3 = nn.Conv3d(self.num_channels, self.num_channels, 3)
            conv3_bn = nn.BatchNorm3d(self.num_channels)
        else:
            conv3 = nn.Conv2d(self.num_channels, self.num_channels, 3)
            conv3_bn = nn.BatchNorm2d(self.num_channels)

        return nn.ModuleList(
                [conv_red, conv_red_bn, conv1, conv1_bn, conv2, conv2_bn, conv3, conv3_bn])

    def make_feature_extraction_model(self, drop_blocks, model_choice):
        if model_choice == 'xception':
            model = xception()

        elif model_choice == 'resnet50' or model_choice == 'resnet18':
            if model_choice == 'resnet50':
                model = torchvision.models.resnet50(pretrained=True)
            if model_choice == 'resnet18':
                model = torchvision.models.resnet18(pretrained=True)
        else:
            raise Exception('Choose valid model, e.g. resnet50')
        # Replace fc
        model = torch.nn.Sequential(*(list(model.children())[:-drop_blocks]))
        return model

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
            for _, param in self.feature_extraction_frames[frame].named_parameters():
                param.requires_grad = False
            # for _, param in self.fc_end.named_parameters():
            #    param.requires_grad = True
        for frame in range(self.num_frames - 1):
            for _, param in self.feature_extraction_flow[frame].named_parameters():
                param.requires_grad = False
            # for _, param in self.fc_end.named_parameters():
            #    param.requires_grad = True

    def forward(self, x):
        y = []
        y_of = []
        image = x["image"]
        optical_flow = x["warp"]

        # Extract features from frames
        for i in range(self.num_frames):
            x_i = image[:, :, i, :, :]
            x_i = x_i.squeeze(dim=1)
            model_i = self.feature_extraction_frames[i]
            y_i = model_i(x_i)
            image_size = y_i.shape[3]
            y.append(y_i.view((-1, self.temporal_encoder_input_channels, image_size, image_size, 1)))

        # Extract features from flow
        for i in range(self.num_frames):
            x_i = optical_flow[:, :, i, :, :]
            x_i = x_i.squeeze(dim=1)
            model_i = self.feature_extraction_flow[i]
            y_i = model_i(x_i)
            image_size = y_i.shape[3]
            y_of.append(y_i.view((-1, self.temporal_encoder_input_channels, image_size, image_size, 1)))

        # conv block for frames
        y_images = self.forward_conv_branch_block(y, self.conv_block_frames)
        y_of = self.forward_conv_branch_block(y_of, self.conv_block_flow)

        # FC layer
        y = y.view(-1, y.shape[1] * y.shape[2] * y.shape[3])
        y = self.fc_end(y)
        return y

    def forward_conv_branch_block(self, y, conv_branch_block):
        image_size = y[0].shape[3]
        y = torch.cat(y, dim=4)
        y = F.relu(conv_branch_block[1](conv_branch_block[0](y)))
        if (y.shape[-1] == 1):
            y = y.view(-1, y.shape[1], y.shape[2], y.shape[3])
        y = F.relu(conv_branch_block[3](conv_branch_block[2](y)))
        y = y.view(-1, self.num_channels, image_size - 2, image_size - 2, self.num_frames - 2)
        if (y.shape[-1] == 1):
            y = y.view(-1, y.shape[1], y.shape[2], y.shape[3])
        y = F.relu(conv_branch_block[5](conv_branch_block[4](y)))
        if (y.shape[-1] == 1):
            y = y.view(-1, y.shape[1], y.shape[2], y.shape[3])
        y = F.relu(conv_branch_block[7](self.conv_branch_block[6](y)))
        return y


if __name__ == '__main__':
    model = OFModel(model_choice='xception', num_freatures=5, dropout=0.0)
    model.train_only_last_layer()
    print(model)