from skimage import io
from torch.utils import data
import torch
import os
import numpy as np


class FaceForensicsVideosDataset(data.Dataset):
    def __init__(self, directories, num_frames, skip_frames=0, overlap=0, transform=None):
        """
        Args:
        directories: directories where image folders are in. Has the following form:  {path: True, path2: False ...}, where True/False implies the label
        num_frames: Number of frames per sample
        skip_frames: How many frames to skip
        overlap: How many frames "neighboring" samples have in common
        """
        self.num_frames = num_frames
        self.skip_frames = skip_frames
        self.transform = transform
        self.overlap = overlap

        self.frame_dir = {}
        x = 0
        for path in directories:
            image_folders = [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]
            for f in image_folders:
                whole_path = path + "/" + f  + "/downsampled"
                n = len([f for f in os.listdir(whole_path) if os.path.isfile(os.path.join(whole_path, f)) and f.endswith(".png")])
                frame_numbers = list(range(n))
                i = 0
                while i < n:
                    for _ in range(self.skip_frames + 1):
                        frames = frame_numbers[i: i + self.get_whole_length() + 2:self.skip_frames + 1]
                        if len(frames) == self.num_frames:
                            self.frame_dir[x] = (whole_path, frames, directories[path])
                            x += 1
                        i += 1
                    i += (self.num_frames - overlap) * (self.skip_frames + 1) - skip_frames -1

    def get_whole_length(self):
        return (self.num_frames -1) * (self.skip_frames + 1)  - 1


    def __len__(self):
        length = len(self.frame_dir)
        return length

    def __getitem__(self, idx):
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
