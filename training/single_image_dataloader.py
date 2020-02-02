"""
Authors: Anna Mittermair
"""



from skimage import io
from torch.utils import data
import torch
import os
import numpy as np
import math


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

        # input: part of cases which should be train and validation set, e.g. 0.8, 0.2
        # returns two lists of indices, one for training and one for test cases

    def get_train_val_lists(self, part_train, part_val):
        label_list = [[0, 0]]
        last_label = (self.frame_dir[0][1] == -1)

        for key in self.frame_dir:
            if (self.frame_dir[key][1] == -1) == last_label:
                label_list[-1][1] += 1
            else:
                label_list.append([key, 0])
                last_label = not last_label

        val_list = []
        train_list = []
        for streak in label_list:
            start = streak[0]
            length = streak[1]

            midpoint = int(start + length * part_train)
            endpoint = math.ceil(midpoint + length * part_val)

            train_part = list(range(start, midpoint))
            val_part = list(range(midpoint, endpoint))
            train_list += train_part
            val_list += val_part

        return train_list, val_list

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
        return {'image'     : image.float() / 255.0,
                'label'     : torch.tensor(label).long()}

if __name__ == '__main__':
    # test /example
    d = [
        "F:/Google Drive/FaceForensics_large/original_sequences/youtube/c40/sequences_299x299_10seq@10frames_skip_5_uniform/train",
        "F:/Google Drive\FaceForensics_large/manipulated_sequences/Deepfakes/c40/sequences_299x299_10seq@10frames_skip_5_uniform/train"]

    test_dataset = FaceForensicsImagesDataset(d,transform=ToTensor())
    print(test_dataset.__len__())
    train_list, val_list = test_dataset.get_train_val_lists(0.9, 0.01)
    print(len(train_list), len(val_list))
    dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=4, shuffle=True,
                                                 num_workers=4)
    for i, sample in enumerate(dataset_loader):
        if i == 0:
            print(sample["label"])
            print(sample["image"].shape)
            #print(sample["label"].shape)