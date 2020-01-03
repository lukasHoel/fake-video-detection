from networks.xception import xception
import torch
import torch.nn as nn
import torchvision

class TemporalEncoder(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes

    Adapted slightly from FaceForensics version: https://github.com/ondyari/FaceForensics/blob/master/classification/network/models.py
    """
    def __init__(self, num_input_images, delta_t=2, feature_dimension=256, model_choice='xception', num_out_classes=2, dropout=0.0):
        super(TemporalEncoder, self).__init__()
        self.model_choice = model_choice
        self.dropout = dropout
        self.num_input_images = num_input_images
        self.delta_t = delta_t
        self.images_in_sequence = 2 * delta_t + 1
        self.num_sequences = num_input_images - 2*delta_t
        self.feature_dimension = feature_dimension
        self.feature_extractor = self.create_feature_extractor(self.model_choice)

        self.temporal_encoder = self.create_temporal_encoder(num_input=self.images_in_sequence,
                                                             channels_per_input=self.feature_dimension)

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
                      kernel_size=3, stride=1, padding=1, bias=True),
            activation_block,
            nn.BatchNorm2d(num_features=input_dim),
            #nn.GroupNorm(num_channels=input_dim, num_groups=int(input_dim/5)),
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim,
                      kernel_size=3, stride=1, padding=1, bias=True),
            activation_block,
            nn.BatchNorm2d(num_features=input_dim),
            #nn.GroupNorm(num_channels=input_dim, num_groups=int(input_dim / 5)),
            #Scale down to channels_per_input channels again (e.g. to 256 channels)
            nn.Conv2d(in_channels=input_dim, out_channels=channels_per_input, kernel_size=1)
        )

        return temporal_encoder

    def create_classifier(self, num_classes, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, int(input_dim/2), bias=True),
            nn.ReLU(),
            nn.Linear(int(input_dim/2), num_classes, bias=True)
        )

    def forward(self, x):
        image_features = [] # image features of every video frame in x independently
        sequence_features = [] # features of every self.delta_t*2 + 1 image features
        predictions = [] # predictions per sequence_feature

        # 1. for every video frame in sequence x: calculate features with self.feature_extractor
        for i in range(self.num_input_images):
            x_i = x[:, :, i, :, :, :]
            x_i = x_i.squeeze(dim=1)
            y_i = self.feature_extractor(x_i)
            image_features.append(y_i)

        # 2. concatenate multiple features together and run a CNN block on it (self.temporal_encoder)
        image_feature_length = len(image_features)
        for i in range(self.num_sequences):
            features = tuple(image_features[i:i+self.images_in_sequence])
            feature_stack = torch.cat(features, 1)
            y_i = self.temporal_encoder(feature_stack)
            sequence_features.append(y_i)

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

        return y

if __name__ == '__main__':
    model = TemporalEncoder(model_choice='xception', num_out_classes=2, dropout=0.0)
    #model.train_only_last_layer()
    #print(model)

    #for name, param in baseline.model.named_parameters():
    #    if param.requires_grad:
    #        print("param: {} requires_grad: {}".format(name, param.requires_grad))