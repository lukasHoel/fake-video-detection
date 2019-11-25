from skimage import io
from torch.utils import data
import torch
import os
import numpy as np


class FaceForensicsImagesDataset(data.Dataset):
    def __init__(self, directories, transform=None):
        """
        Args:
        directories: List of paths where the images for the dataset are
            Example path: "... /manipulated_sequences/Face2Face/c40/sequences.
            In this directory, there needs to be a folder /sequences containing folders of sequences with the png images which will be used.
        """
        self.transform = transform
        mapping_dict = {}
        self.dataset_length = 0

        self.frame_dir = {}
        counter = 0
        for path in directories:
            # Get all folders with videos in the directory at path
            video_folders = [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]
            for f in video_folders:
                # process video name to know how it was generated
                name_split = f.split("_")
                if len(name_split) == 1:
                    name_split = [-1] + name_split  # length 1 indicates original -> set actor to -1
                actor = name_split[0]
                original = name_split[1]

                # Iterate through all sequences
                sequence_folders = [os.path.join(path, f, x) for x in os.listdir(os.path.join(path, f)) if
                                    not os.path.isfile(os.path.join(path, f, x))]
                for s in sequence_folders:
                    # Discard empty sample folders
                    images = [os.path.join(s, x) for x in os.listdir(s) if os.path.isfile(os.path.join(s, x)) and x.endswith(".png")]
                    for i in images:
                        key = counter
                        counter += 1
                        self.frame_dir[key] = (i, actor, original)
        self.dataset_length = counter

    # number of samples in the dataset
    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        """
        Gets items with the id idx or idx_* from self.frame_dir, loads and returns them.
        Returns one sample of the following form:
        {    sample = list of numpy arrays consisting of num_frames frames of downsampled images from one video,
             label = list of label for each sample, original = 1, fake = 0}

        Every idx is connected to one sample, by being the key of the sample in self.frame_dir.
        This mapping from ids to samples is not random, but is caused by to the order of directories and frames in the video.
        Therefore samples need to be retrieved in a randomized order.
       """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        whole_path, actor, original = self.frame_dir[idx]
        label = actor == -1   # Append label 1 if actor==-1 (so if video not fake)
        image = io.imread(whole_path)
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        return {'image'     : image,
                'label'     : label}

if __name__ == '__main__':
    # test /example
    d = ["C:/Users/admin/Desktop/FaceForensics/manipulated_sequences/Face2Face/c40/sequences"]
    test_dataset = FaceForensicsImagesDataset(d,transform=ToTensor())
    dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=4, shuffle=True,
                                                 num_workers=4)
    for i, sample in enumerate(dataset_loader):
        if i == 0:
            print(sample["label"])
            print(sample["image"].shape)
            #print(sample["label"].shape)