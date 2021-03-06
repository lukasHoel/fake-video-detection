"""
Authors: Anna Mittermair and Lukas Hoellein
"""

from utils.warp_image_farneback import warp_from_images
from skimage import io
from torch.utils import data
import torch
import os
import numpy as np
import math
from tqdm.auto import tqdm
from torchvision import transforms

class FaceForensicsVideosDataset(data.Dataset):
    def __init__(self, directories, num_frames, generate_coupled=False, transform=None, max_number_videos_per_directory=-1, max_number_sequences_per_video=-1, calculateOpticalFlow=True, verbose=False, caching=True):
        """
        Args:
        directories: List of paths where the images for the dataset are
            Example path: "... /manipulated_sequences/Face2Face/c40/sequences.
            In this directory, there needs to be a folder /sequences containing folders of sequences with the png images which will be used.
        generate_coupled: Groups all sequences of fakes made from the same original and the original together in one group of samples that can be retrieved at once.

        max_number_videos_per_directory: after how many videos to stop searching, e.g. useful if directory contains 1000 videos, but only want to train on the first 100.
        max_number_sequences_per_video: after how many sequences to stop searching, e.g. useful if video is split in 10 sequences, but we only want to train on the first 2.
        verbose: whether or not to print when a new image is loaded (and opt-flow is calculated)
        calculateOpticalFlow: Whether or not to calculate it when loading an image
        caching: If set to true, will save the image (and warp) in a dictionary on RAM when first requesting an item. This can speed up the second epoch of training at the cost of high RAM memory consumption.
        """
        self.transform = transform
        self.generate_coupled = generate_coupled
        mapping_dict = {}
        self.dataset_length = 0
        self.num_videos = 0
        self.num_frames=num_frames
        self.max_number_videos_per_directory = max_number_videos_per_directory
        self.max_number_sequences_per_video = max_number_sequences_per_video
        self.calculateOpticalFlow = calculateOpticalFlow
        self.verbose = verbose
        self.caching = caching
        self.frame_dir = {}
        counter = 0
        number_directories = 0
        for path in directories:
            # Get all folders with videos in the directory at path
            number_directories += 1
            print("Loading directory {}/{}: {}".format(number_directories, len(directories), path))

            number_videos_for_directory = 0
            video_folders = [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]
            for f in tqdm(video_folders):
                if self.max_number_videos_per_directory >= 0 and number_videos_for_directory >= self.max_number_videos_per_directory:
                    print("Reached maximum number of videos per directory ({}), will skip the rest.".format(number_videos_for_directory))
                    break
                self.num_videos += 1
                number_videos_for_directory += 1
                # process video name to know how it was generated
                name_split = f.split("_")
                if len(name_split) == 1:
                    name_split = [-1] + name_split  # length 1 indicates original -> set actor to -1
                actor = name_split[0]
                original = name_split[1]

                # Iterate through all sequences
                sequence_folders = [os.path.join(path, f, x) for x in os.listdir(os.path.join(path, f)) if
                                    not os.path.isfile(os.path.join(path, f, x))]
                number_sequences_for_video = 0
                for s in sequence_folders:
                    if self.max_number_sequences_per_video >= 0 and number_sequences_for_video >= self.max_number_sequences_per_video:
                        if self.verbose:
                            print("Reached maximum number of sequences per video ({}), will skip the rest.".format(
                                number_sequences_for_video))
                        break
                    number_sequences_for_video += 1

                    # Discard empty sample folders
                    n = len([x for x in os.listdir(s) if os.path.isfile(os.path.join(s, x)) and x.endswith(".png")])
                    if n < self.num_frames:
                        # print(sequence_length, n)
                        continue

                    # Generate dict_keys of form f(original)_index for all videos
                    # * f(original) is a bijective mapping of original ids to the values 0 to len(dataset)
                    #   The mapping is necessary to later retrieve the samples by an id between 0 and len(dataset),
                    #   otherwise some ids would not have a corresponding sample
                    # * index is the number of samples generated from the same original that have been added before.
                    #   This gives all samples generated from the same original and the original different ids with the same beginning as an identifier
                    label = (actor == -1)
                    # get all images at whole_path/
                    image_names = [f for f in os.listdir(s) if
                                   os.path.isfile(os.path.join(s, f)) and f.endswith(".png")]
                    image_names = image_names[0:self.num_frames]

                    # Read all image file names into a list for lazy-loading upon first get_item request
                    image_paths = []
                    for name in image_names:
                        image_paths.append(s + "/" + name)

                    key = counter
                    counter += 1
                    self.frame_dir[key] = (False, image_paths, label, None)
        self.dataset_length = counter

    # number of samples in the dataset
    def __len__(self):
        return self.dataset_length

    def get_num_videos(self):
        return self.num_videos

    # input: part of cases which should be train and validation set, e.g. 0.8, 0.2
    # returns two lists of indices, one for training and one for test cases
    def get_train_val_lists(self, part_train, part_val):
        label_list = [[0, 0]]
        last_label = self.frame_dir[0][2]
        for key in self.frame_dir:
            if self.frame_dir[key][2] == last_label:
                label_list[-1][1] += 1
            else:
                label_list.append([key, 0])
                last_label = not last_label
        label_list[-1][1] += 1
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

    def __getitem__(self, idx):
        """
        Gets items with the id idx or idx_* from self.frame_dir, loads and returns them.
        Returns one sample of the following form:
        {    samples = list of numpy arrays consisting of num_frames frames of downsampled images from one video,
             labels = list of label for each sample, original = 1, fake = 0
             original_id = id of original video}
        If generate_coupled = False:
            Every idx is connected to one sample, by being the key of the sample in self.frame_dir.
            This mapping from ids to samples is not random, but is caused by to the order of directories and frames in the video.
            Therefore samples need to be retrieved in a randomized order.
        If generate_coupled = True:
            Every idx is connected to several samples being one original video and all fake ones generated by it that
            are present in the dataset. idx_sampleindex is being the key of the samples in self.frame_dir.
       """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        is_loaded, sequence, label, warp = self.frame_dir[idx]

        # lazy-load sequence when it is requested for the first time
        if not is_loaded:
            this_sample = []
            if self.verbose:
                print("loading new sequence")
                for name in tqdm(sequence):
                    this_sample.append(io.imread(name))
            else:
                for name in sequence: # iterate without tqdm(...)
                    this_sample.append(io.imread(name))
            sequence = np.stack(this_sample)
            if self.calculateOpticalFlow:
                warp = []
                n = sequence.shape[0]
                center_image = sequence[n//2]
                if self.verbose:
                    print("calculating and loading optical-flow warp")
                    for i in tqdm(range(n)):
                        warp.append(warp_from_images(sequence[i], center_image))
                else:
                    for i in range(n): # iterate without tqdm(...)
                        warp.append(warp_from_images(sequence[i], center_image))
                warp = np.stack(warp)

            if self.caching:
                if self.verbose:
                    print("Save calculated results into cache")
                self.frame_dir[idx] = (True, sequence, label, warp)

        # todo why do we need these two lines? in the models, we squeeze the extra dimension everytime?
        samples = np.stack([sequence])
        warps = np.stack([warp])

        sample = {"image": samples, "label": label, "warp": warps}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        # normalize similar to FaceForensics paper: https://github.com/ondyari/FaceForensics/blob/master/classification/dataset/transform.py
        self.normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)

    def __call__(self, sample):
        samples, labels, warps = sample["image"], sample["label"], sample["warp"]

        # swap color axis because
        # numpy image: num_frames x H x W x C
        # torch image: num_frames x C X H X W
        samples = torch.from_numpy(samples.transpose((0, 1, 4, 2, 3))).float() / 255.0

        # normalize
        for b in range(samples.shape[0]):
            for s in range(samples.shape[1]):
                samples[b][s] = self.normalize(samples[b][s])

        labels = torch.tensor(labels).long()

        result = {"image": samples,
                  "label": labels}

        if warps is not None and None not in warps:
            warps = torch.from_numpy(warps.transpose((0, 1, 4, 2, 3))).float() / 255.0
            for b in range(warps.shape[0]):
                for s in range(warps.shape[1]):
                    warps[b][s] = self.normalize(warps[b][s])
            result["warp"] = warps

        return result

def my_collate(batch):
    data = np.concatenate([b["image"] for b in batch], axis=0)
    targets = [b["label"] for b in batch]
    warp = [b["warp"] for b in batch]
    sample = {"image": data, "label": targets, "warp": warp}
    return sample


if __name__ == '__main__':
    # test /example
    d = [
        "F:/Google Drive/FaceForensics_large/original_sequences/youtube/c40/sequences_299x299_10seq@10frames_skip_5_uniform/train",
        "F:/Google Drive\FaceForensics_large/manipulated_sequences/Deepfakes/c40/sequences_299x299_10seq@10frames_skip_5_uniform/train"]
    test_dataset = FaceForensicsVideosDataset(d, generate_coupled=False, num_frames=10, transform=ToTensor(),
                                              max_number_videos_per_directory=4, verbose=False)
    print(test_dataset.__len__())
    train_list, val_list = test_dataset.get_train_val_lists(0.8, 0.2)
    print(len(train_list), len(val_list))

    from torch.utils.data.sampler import SubsetRandomSampler

    train_sampler = SubsetRandomSampler(train_list)
    valid_sampler = SubsetRandomSampler(val_list)

    # Should set num_workers=0, otherwise the caching in the dataset does not work... but why?
    train_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                               sampler=train_sampler,
                                               num_workers=1)

    validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                    sampler=valid_sampler,
                                                    num_workers=1)

    '''
    sum_0 = 0
    sum_1 = 0
    for v in list(validation_loader):

        if v["label"].data.cpu().numpy()[0] == 0:
            sum_0 += 1
        else:
            sum_1 += 1

    print("validation loader has {} zeros and {} ones out of total {}".format(sum_0, sum_1, len(validation_loader)))

    sum_0 = 0
    sum_1 = 0
    for v in list(train_loader):

        if v["label"].data.cpu().numpy()[0] == 0:
            sum_0 += 1
        else:
            sum_1 += 1

    print("train loader has {} zeros and {} ones out of total {}".format(sum_0, sum_1, len(train_loader)))
    '''

    for sample in train_loader:
        xb = sample["image"]
        print(xb.shape)
        for idx in range(xb.shape[2]):
            import matplotlib.pyplot as plt

            plt.imshow(xb[0][0][idx].cpu().numpy().transpose((1,2,0)))
            plt.show()

    '''
    dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=4, shuffle=True,
                                                 collate_fn=my_collate,  # use custom collate function here
                                                 pin_memory=True)

    for i, sample in enumerate(dataset_loader):
        print(sample["label"])
    '''


