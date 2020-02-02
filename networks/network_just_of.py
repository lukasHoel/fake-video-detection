from networks.xception import xception
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch
import numpy as np


class Just_OF_Model(nn.Module):
    """
    """

    def __init__(self, model_choice='xception', num_frames=5, drop_blocks=8, image_shape=299):
        super(Just_OF_Model, self).__init__()
        self.num_frames = num_frames
        self.model_choice = model_choice
        self.feature_extraction_stage = []
        self.combined_conv_stage = []
        first_stage_out_ftrs = 0

        num_3d_layers = self.num_frames // 2

        # Feature Extraction stage
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

            # Drop last num_blocks blocks
            model = torch.nn.Sequential(*(list(model.children())[:-drop_blocks]))
            test = torch.Tensor(np.ones((1, 3, image_shape, image_shape)))
            output_shape = model(test).shape
            output_image_size = output_shape[2]
            first_stage_out_ftrs = output_shape[1]

            self.feature_extraction_stage.append(model)
        self.feature_extraction_stage = nn.ModuleList(self.feature_extraction_stage)
            
        self.temporal_encoder_input_channels = first_stage_out_ftrs
        self.num_channels = 64
        self.temporal_encoder_output_channels = (output_image_size - 3 * 2) ** 2 * self.num_channels

        # Second stage with conv layers
        self.combined_conv_stage = []
        
        # Channel reduction convolution
        self.combined_conv_stage.append(nn.Conv3d(self.temporal_encoder_input_channels, self.num_channels, 1))
        self.combined_conv_stage.append(nn.BatchNorm3d(self.num_channels))

        # Conv Layers
        self.combined_conv_stage.append(nn.Conv3d(self.num_channels, self.num_channels, 3))
        self.combined_conv_stage.append(nn.BatchNorm3d(self.num_channels))

        if num_3d_layers > 1:
            self.combined_conv_stage.append(nn.Conv3d(self.num_channels, self.num_channels, 3))
            self.combined_conv_stage.append(nn.BatchNorm3d(self.num_channels))
        else:
            self.combined_conv_stage.append(nn.Conv2d(self.num_channels, self.num_channels, 3))
            self.combined_conv_stage.append(nn.BatchNorm2d(self.num_channels))
        if num_3d_layers > 2:
            self.combined_conv_stage.append(nn.Conv3d(self.num_channels, self.num_channels, 3))
            self.combined_conv_stage.append(nn.BatchNorm3d(self.num_channels))
        else:
            self.combined_conv_stage.append(nn.Conv2d(self.num_channels, self.num_channels, 3))
            self.combined_conv_stage.append(nn.BatchNorm2d(self.num_channels))

        self.combined_conv_stage = nn.ModuleList(self.combined_conv_stage)

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
        # Stage-1: freeze all the layer
        for frame in range(self.num_frames - 1):
            for _, param in self.feature_extraction_stage[frame].named_parameters():
                param.requires_grad = False
            # for _, param in self.fc_end.named_parameters():
            #    param.requires_grad = True

    def forward(self, x):
        y_of = []
        
        # Optical Flow is called warp in dict to be able to use the same solver
        optical_flow = x["warp"]
        
        # Extract features (first stage)
        for i in range(self.num_frames):
            x_i = optical_flow[:, :, i, :, :]
            x_i = x_i.squeeze(dim=1)
            model_i = self.feature_extraction_stage[i]
            y_i = model_i(x_i)
            image_size = y_i.shape[3]
            y_of.append(y_i.view((-1, self.temporal_encoder_input_channels, image_size, image_size, 1)))

        # Second stage
        
        # Concatenate features along new dimension
        y = torch.cat(y_of, dim=4)
        
        # CNN channel reduction
        y = F.relu(self.combined_conv_stage[1](self.combined_conv_stage[0](y)))

        # CNN block 1
        if (y.shape[-1] == 1):
            y = y.view(-1, y.shape[1], y.shape[2], y.shape[3])
        y = F.relu(self.combined_conv_stage[3](self.combined_conv_stage[2](y)))

        # CNN block 2
        y = y.view(-1, self.num_channels, image_size - 2, image_size - 2, self.num_frames - 2)
        if (y.shape[-1] == 1):
            y = y.view(-1, y.shape[1], y.shape[2], y.shape[3])
        y = F.relu(self.combined_conv_stage[5](self.combined_conv_stage[4](y)))

        # CNN block 3
        if (y.shape[-1] == 1):
            y = y.view(-1, y.shape[1], y.shape[2], y.shape[3])
        y = F.relu(self.combined_conv_stage[7](self.combined_conv_stage[6](y)))

        # FC layer
        y = y_of.view(-1, y.shape[1] * y.shape[2] * y.shape[3])
        y = self.fc_end(y)
        return y


if __name__ == '__main__':
    of_model = Just_OF_Model()
    of_model.train_only_last_layer()
    print(of_model)