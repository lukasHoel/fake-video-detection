from skimage import io
from torch.utils import data
import torch
import os
import numpy as np


class FaceForensicsVideosDataset(data.Dataset):
    def __init__(self, directories, num_frames, skip_frames=0, overlap=0, transform=None):
        """
        Args:
        directories: directories where image folders are in. Has the following form:  {path: label, path2: label ...}.
            All images belonging to one path need to have the same label.
            Example path: "... /manipulated_sequences/Face2Face/c40/images".
            In this directory, there needs to be a folder for each video, containing a folder downsampled containing the png images which will be used.
        num_frames: Number of frames per sample.
        skip_frames: How many frames to skip between two frames in one sample.
        overlap: How many frames "neighboring" samples have in common.

        Example: num_frames = 3, skip = 2, overlap = 1

        Frames in sample 1: 1, 4, 7    (skipped 2, 3 and 5, 6)
        Frames in sample 2: 2, 5, 8     ...
        Frames in sample 3: 3, 6, 9

        Frames in sample 4: 7, 10, 13  (overlaps with sample 1 on 1 frame (number 7)
        Frames in sample 5: 8, 11, 14  (overlaps with sample 2 on 1 frame (number 8)
        Frames in sample 6: 9, 12, 15  (overlaps with sample 3 on 1 frame (number 9)

        Frames in sample 7: 13, 11, 14  (overlaps with sample 4 on 1 frame (number 13)
        ...

        For each folder of downsampled images, the frame numbers for all samples are generated and saved in self.frame_dir.
        Each sample in this directory is a tuple (path to folder with the images, list of sample frames, label)
        with the key being the number of the sample (numberd chronologically in the order they were generated)
        """
        self.num_frames = num_frames
        self.skip_frames = skip_frames
        self.transform = transform
        self.overlap = overlap

        self.frame_dir = {}
        x = 0
        for path in directories:
            # Get all folders with images in the directory at path
            image_folders = [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]
            for f in image_folders:
                whole_path = path + "/" + f  + "/downsampled"
                n = len([f for f in os.listdir(whole_path) if os.path.isfile(os.path.join(whole_path, f)) and f.endswith(".png")])

                # Generate frame numbers for this video, according to num_parameters, skip and overlap
                frame_numbers = list(range(n))
                i = 0
                while i < n:
                    for _ in range(self.skip_frames + 1):
                        # Frame numbers for new sample:
                        frames = frame_numbers[i: i + self.get_whole_length() + 2:self.skip_frames + 1]
                        if len(frames) == self.num_frames:
                            # add sample to_frame_dir
                            self.frame_dir[x] = (whole_path, frames, directories[path])
                            x += 1
                        i += 1
                    i += (self.num_frames - overlap) * (self.skip_frames + 1) - skip_frames -1

    def get_whole_length(self):
        return (self.num_frames -1) * (self.skip_frames + 1)  - 1

    # number of samples in the dataset
    def __len__(self):
        length = len(self.frame_dir)
        return length

    def __getitem__(self, idx):
        """
        Returns one sample of the following form:
        {    image =(numpy array consisting of num_frames frames of downsampled images from one video),
             label = (assigned label)}
        Every idx is connected to one sample, by being the key of the sample in self.frame_dir.
        This mapping from ids to samples is not random, but corresponds to the order of directories and frames in the video.
        Therefore samples need to be retrieved in a randomized order.
       """
        path, frames, label = self.frame_dir[idx]
        images = []
        for f in frames:
            image_name = "{:04d}_-1x120.png".format(f)
            images.append(io.imread(path + "/" + image_name))

        # all as a numpy array
        image_matrix = np.stack(images)

        sample = {'images': image_matrix, 'label': label }

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        images, label = sample['images'], sample['label']

        # swap color axis because
        # numpy image: num_frames x H x W x C
        # torch image: num_frames x C X H X W
        images = images.transpose((0, 3, 1, 2))
        return {'image': torch.from_numpy(images),
                'landmarks': torch.from_numpy(label)}


# test /example
d = {"/home/anna/Desktop/Uni/WiSe19/DL4CV/data/FaceForensics/d1/manipulated_sequences/Face2Face/c40/images": True,"/home/anna/Desktop/Uni/WiSe19/DL4CV/data/FaceForensics/d1/manipulated_sequences/Face2Face/c40/images":False }
test_dataset = FaceForensicsVideosDataset(d, num_frames=4, skip_frames=5, transform=ToTensor)
dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)
