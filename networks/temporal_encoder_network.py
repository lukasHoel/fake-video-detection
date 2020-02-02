"""
Architecture of the temporal encoder network:  feature extraction for every image frame independently followed by concatenation of multiple image features in a temporal encoder block before binary classification. The temporal encoder block concatenates an amount delta_t of image features and forwards them through n CNN blocks (same convolution, ReLU, batchnorm). This is done multiple times with the same layers until all input images from one sequence are processed.  Finally, we combine everything with fully-connected layers.

The network can be configured to use additional warping inputs in sequences via the 'useOpticalFlow' param.

Author: Lukas Hoellein
"""

from networks.xception import xception
import torch
import torch.nn as nn
import torchvision
from time import time

class TemporalEncoder(nn.Module):
    """
    Architecture of the temporal encoder network:  feature extraction for every image frame independently followed by concatenation of multiple image features in a temporal encoder block before binary classification. The temporal encoder block concatenates an amount delta_t of image features and forwards them through n CNN blocks (same convolution, ReLU, batchnorm). This is done multiple times with the same layers until all input images from one sequence are processed.  Finally, we combine everything with fully-connected layers.

    The network can be configured to use additional warping inputs in sequences via the 'useOpticalFlow' param.

    Author: Lukas Hoellein
    """
    def __init__(self, num_input_images, delta_t=2, feature_dimension=64, temporal_encoder_depth=5,
                 model_choice='xception', num_out_classes=2, dropout=0.0,
                 useOpticalFlow=True):
        """
        Architecture of the temporal encoder network:  feature extraction for every image frame independently followed by concatenation of multiple image features in a temporal encoder block before binary classification. The temporal encoder block concatenates an amount delta_t of image features and forwards them through n CNN blocks (same convolution, ReLU, batchnorm). This is done multiple times with the same layers until all input images from one sequence are processed.  Finally, we combine everything with fully-connected layers.

        The network can be configured to use additional warping inputs in sequences via the 'useOpticalFlow' param.

        Author: Lukas Hoellein

        Parameters
        ----------
        num_input_images: how many images are in one sequence
        delta_t: how many images to choose before and after the current image for grouping in the temporal encoder
        feature_dimension: how many features to extract per image
        temporal_encoder_depth: how many iterations of the temporal encoder CNN should we do
        model_choice: which pretrained model to use for feature extraction
        num_out_classes: e.g. binary classification
        dropout: how much to use
        useOpticalFlow: True/False: use as additional input or not (here optical flow actually means warped images, see report)
        """
        super(TemporalEncoder, self).__init__()
        self.model_choice = model_choice
        self.dropout = dropout
        self.num_input_images = num_input_images
        self.delta_t = delta_t
        self.temporal_encoder_depth = temporal_encoder_depth
        self.images_in_sequence = 2 * delta_t + 1
        if useOpticalFlow:
            self.images_in_sequence *= 2
        self.num_sequences = num_input_images - 2*delta_t
        self.feature_dimension = feature_dimension
        self.useOpticalFlow = useOpticalFlow

        self.feature_extractor = self.create_feature_extractor(self.model_choice)
        self.temporal_encoder = self.create_temporal_encoder(num_input=self.images_in_sequence,
                                                             channels_per_input=self.feature_dimension,
                                                             depth=temporal_encoder_depth)

        # 10*10 comes from spatial domain resoltuion from xception net
        # todo not hardcode!
        self.single_classifier = self.create_classifier(num_classes=num_out_classes,
                                                        input_dim=self.feature_dimension*10*10)
        self.overall_classifier = self.create_classifier(num_classes=num_out_classes,
                                                         input_dim=num_out_classes*self.num_sequences)

    def create_feature_extractor(self, model_choice="xception"):
        if model_choice == 'xception':
            feature_extractor = xception()
            for i, param in feature_extractor.named_parameters():
                param.requires_grad = False
            # Remove fc

            removed = list(feature_extractor.children())[:-1]
            feature_extractor = nn.Sequential(
                *removed,
                 nn.Conv2d(in_channels=2048, out_channels=self.feature_dimension, kernel_size=1))

        # todo for resnet: 1x1 conv to have less features how?
        elif model_choice == 'resnet50' or model_choice == 'resnet18':
            if model_choice == 'resnet50':
                feature_extractor = torchvision.models.resnet50(pretrained=True)
            if model_choice == 'resnet18':
                feature_extractor = torchvision.models.resnet18(pretrained=True)

            for i, param in feature_extractor.named_parameters():
                param.requires_grad = False
            # Remove fc
            removed = list(feature_extractor.children())[:-1]
            feature_extractor = nn.Sequential(*removed)
        else: # todo add efficient net via: https://github.com/lukemelas/EfficientNet-PyTorch
            raise Exception('Choose valid model, e.g. resnet50')

        return feature_extractor

    def create_temporal_encoder(self, num_input, channels_per_input, depth):
        input_dim = num_input*channels_per_input

        if self.dropout > 0:
            activation_block = nn.Sequential(
                nn.ReLU(),
                nn.Dropout2d(p=self.dropout)
            )
        else:
            activation_block = nn.ReLU()

        convs = []
        for i in range(depth):
            convs.append(nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=input_dim,
                          kernel_size=3, stride=1, padding=1, bias=True),
                activation_block,
                # nn.GroupNorm(num_channels=input_dim, num_groups=int(input_dim/5)),
                nn.BatchNorm2d(num_features=input_dim)
            ))

        temporal_encoder = nn.Sequential(
            *convs,
            #Scale down to channels_per_input channels again (e.g. to 256 channels)
            nn.Conv2d(in_channels=input_dim, out_channels=channels_per_input, kernel_size=1),
            activation_block
        )

        return temporal_encoder

    def create_classifier(self, num_classes, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, int(input_dim/2), bias=True),
            nn.ReLU(),
            nn.Linear(int(input_dim/2), num_classes, bias=True)
        )

    def forward(self, x):
        # by convention (see also solver) this is how the input is delivered depending on optical flow available
        if self.useOpticalFlow:
            images = x["image"]
            warps = x["warp"]
        else:
            images = x

        sequence_features = [] # features of every self.delta_t*2 + 1 image (and flow) features
        predictions = [] # predictions per sequence_feature

        #start = time()
        # 1.a for every video frame in sequence x: calculate features with self.feature_extractor

        # images has dimension: (batch x 1 x sequence_length x C x W x H)
        # Concatenate batch + sequence for fast feature extraction without any for loops! (vectorized impl)

        images = images.squeeze(dim=1) # remove dim=(1)
        b, s, c, w, h = images.shape
        images = images.view(-1, c, w, h)
        images = self.feature_extractor(images)
        _, c, w, h = images.shape
        images = images.view(b, s, c, w, h)

        # 1.b if optical flow is enabled: calculate image features of warped image in each sequence
        if self.useOpticalFlow:

            # warp has dimension: (batch x 1 x sequence_length x C x W x H)
            # Concatenate batch + sequence for fast feature extraction without any for loops! (vectorized impl)
            warps = warps.squeeze(dim=1)  # remove dim=(1)
            b, s, c, w, h = warps.shape
            warps = warps.view(-1, c, w, h)
            warps = self.feature_extractor(warps)
            _, c, w, h = warps.shape
            warps = warps.view(b, s, c, w, h)
        #print("Feature extraction image + warp took: {}".format(time() - start))

        #start = time()
        # 2. concatenate multiple features (normal + flow) together and run a CNN block on it (self.temporal_encoder)
        for i in range(self.num_sequences):

            if self.useOpticalFlow:
                img_features = images[:, i:i + self.images_in_sequence // 2, :, :, :]
                warp_features = warps[:, i:i + self.images_in_sequence // 2, :, :, :]
                features = torch.cat((img_features, warp_features), 1)
            else:
                features = images[:, i:i + self.images_in_sequence, :, :, :]

            b, s, c, w, h = features.shape
            features = features.view(b, s*c, w, h)
            y_i = self.temporal_encoder(features)
            sequence_features.append(y_i)
        #print("Temporal encoder took: {}".format(time() - start))

        #start = time()
        # 3. for every temporal_encoder output: run FC layer and do classification
        for i in range(len(sequence_features)):
            batch_size = sequence_features[i].shape[0]
            features = sequence_features[i].view(batch_size, -1) # flatten for fully connected
            y_i = self.single_classifier(features)
            predictions.append(y_i)

        # 4. calculate final output as follows:
                # as soon as one output of the temporal_encoder was fake --> fake
                # or: majority vote, only if >= 50% say that it was fake --> fake
                # or: average!
                # or: even another metric... (make this configurable!)
                # or: learn it!
        predictions_stack = torch.cat(tuple(predictions), 1)
        y = self.overall_classifier(predictions_stack)
        #print("FC layer took: {}".format(time() - start))

        return y
