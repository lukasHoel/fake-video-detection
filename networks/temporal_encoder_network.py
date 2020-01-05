from networks.xception import xception
from utils.warp_image_farneback import warp_from_images
import torch
import torch.nn as nn
import torchvision

class TemporalEncoder(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes

    Adapted slightly from FaceForensics version: https://github.com/ondyari/FaceForensics/blob/master/classification/network/models.py
    """
    def __init__(self, num_input_images, delta_t=2, feature_dimension=64,
                 model_choice='xception', num_out_classes=2, dropout=0.0,
                 useOpticalFlow=True):
        super(TemporalEncoder, self).__init__()
        self.model_choice = model_choice
        self.dropout = dropout
        self.num_input_images = num_input_images
        self.delta_t = delta_t
        self.images_in_sequence = 2 * delta_t + 1
        self.num_sequences = num_input_images - 2*delta_t
        self.feature_dimension = feature_dimension
        self.useOpticalFlow = useOpticalFlow

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
        else: # todo add efficient net via: https://github.com/lukemelas/EfficientNet-PyTorch
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
        flow_features = [] # if optical flow shall be calculated: save it's image features from warped image in this list
        sequence_features = [] # features of every self.delta_t*2 + 1 image features
        predictions = [] # predictions per sequence_feature

        print("Start forward pass")
        # 1.a for every video frame in sequence x: calculate features with self.feature_extractor
        for i in range(self.num_input_images):
            x_i = x[:, :, i, :, :, :]
            x_i = x_i.squeeze(dim=1)
            y_i = self.feature_extractor(x_i)
            image_features.append(y_i)
        print("Image feature extraction finished")

        # 1.b if optical flow is enabled: calculate optical flow to center of sequence from each image
        # and save image features of warped image
        if self.useOpticalFlow:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            center_image = x[:, :, self.num_input_images//2, :, :, :]
            center_image = center_image.squeeze(dim=1)
            center_image = center_image.data.cpu().numpy()

            # TODO need to loop through batch?
            # TODO order of numpy array must be HxWxC but currently is Cx?x? --> must also reverse this back...
            

            for i in range(self.num_input_images):
                if i == self.num_input_images//2:
                    pass
                    # TODO do we really need to calculate this flow?
                    # TODO I guess not... how to make sure that this can be skipped in step 2 when selecting from flow_features?
                x_i = x[:, :, i, :, :, :]
                x_i = x_i.squeeze(dim=1)

                warp = warp_from_images(x_i.data.cpu().numpy(), center_image)
                warp =  torch.from_numpy(warp).float().to(device)
                y_i = self.feature_extractor(warp)
                flow_features.append(y_i)
        print("Flow feature extraction finished")

        # 2. concatenate multiple features (normal + flow) together and run a CNN block on it (self.temporal_encoder)
        for i in range(self.num_sequences):
            if self.useOpticalFlow:
                features = tuple(image_features[i:i+self.images_in_sequence]+flow_features[i:i+self.images_in_sequence])
            else:
                features = tuple(image_features[i:i+self.images_in_sequence])

            feature_stack = torch.cat(features, 1)
            y_i = self.temporal_encoder(feature_stack)
            sequence_features.append(y_i)
        print("Temporal encoding finished")

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
        print("Forward pass finished")

        return y