from networks.xception import xception
import torch.nn as nn
import torchvision

class TemporalEncoder(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes

    Adapted slightly from FaceForensics version: https://github.com/ondyari/FaceForensics/blob/master/classification/network/models.py
    """
    def __init__(self, num_input_images, model_choice='xception', num_out_classes=2, dropout=0.0):
        super(TemporalEncoder, self).__init__()
        self.model_choice = model_choice
        self.dropout = dropout
        self.num_input_images = num_input_images
        self.feature_extractor = self.create_feature_extractor(self.model_choice)

        # todo make channels_per_input configurable from last layer in feature_extractor
        self.temporal_encoder = self.create_temporal_encoder(num_input=num_input_images, channels_per_input=2048)

    def create_feature_extractor(self, model_choice="xception"):
        if model_choice == 'xception':
            feature_extractor = xception()
            # Remove fc
            removed = list(feature_extractor.children())[:-1]
            feature_extractor = nn.Sequential(*removed)

        elif model_choice == 'resnet50' or model_choice == 'resnet18':
            if model_choice == 'resnet50':
                feature_extractor = torchvision.models.resnet50(pretrained=True)
            if model_choice == 'resnet18':
                feature_extractor = torchvision.models.resnet18(pretrained=True)

            # Remove fc
            removed = list(feature_extractor.children())[:-1]
            feature_extractor = nn.Sequential(*removed)
        else:
            raise Exception('Choose valid model, e.g. resnet50')

        return feature_extractor

    def create_temporal_encoder(self, num_input, channels_per_input):
        input_dim = num_input*channels_per_input

        if self.dropout > 0:
            activation_block = nn.Sequential(
                nn.ReLU(),
                nn.Dropout2d(p=self.dropout)
            )
        else:
            activation_block = nn.ReLU()

        temporal_encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim,
                      kernel_size=3, stride=1, bias=True),
            activation_block,
            nn.GroupNorm(num_channels=input_dim, num_groups=int(input_dim/5)),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim,
                      kernel_size=3, stride=1, bias=True),
            activation_block,
            nn.GroupNorm(num_channels=input_dim, num_groups=int(input_dim / 5))
        )

        return temporal_encoder

    def forward(self, x):

        print(x.shape)

        # x is sequence of length n as input (how to get single image from sequence?)
        for i in range(self.num_input_images):
            x_i = x[:, :, i, :, :, :]
            x_i = x_i.squeeze(dim=1)
            y_i = self.feature_extractor(x_i)

            print(y_i.shape)

            #y_i = y_i.flatten(start_dim=1)
            #y.append(y_i)

            images = ...
            targets = ...

        # 1. for every image in sequence: calculate features with self.feature_extractor
                # todo do I need to loop or does it work vectorized?
                # save the features in an array
        # 2. concatenate multiple features together and run a CNN block on it (self.temporal_encoder)
                # we now have multiple outputs of this kind
        # 3. for every temporal_encoder output: run FC layer and do classification
        # 4. calculate final output as follows:
                # as soon as one output of the temporal_encoder was fake --> fake
                # or: majority vote, only if >= 50% say that it was fake --> fake
                # or: even another metric... (make this configurable!)

        x = self.model(x)
        return x

if __name__ == '__main__':
    model = TemporalEncoder(model_choice='xception', num_out_classes=2, dropout=0.0)
    #model.train_only_last_layer()
    #print(model)

    #for name, param in baseline.model.named_parameters():
    #    if param.requires_grad:
    #        print("param: {} requires_grad: {}".format(name, param.requires_grad))