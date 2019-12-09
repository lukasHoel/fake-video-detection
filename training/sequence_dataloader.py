from skimage import io
from torch.utils import data
import torch
import os
import numpy as np
import math


class FaceForensicsVideosDataset(data.Dataset):
    def __init__(self, directories, sequence_length, generate_coupled=False, transform=None):
        """
        Args:
        directories: List of paths where the images for the dataset are
            Example path: "... /manipulated_sequences/Face2Face/c40/sequences.
            In this directory, there needs to be a folder /sequences containing folders of sequences with the png images which will be used.
        generate_coupled: Groups all sequences of fakes made from the same original and the original together in one group of samples that can be retrieved at once.
        """
        self.transform = transform
        self.generate_coupled = generate_coupled
        mapping_dict = {}
        self.dataset_length = 0
        self.num_videos = 0

        self.frame_dir = {}
        counter = 0
        for path in directories:
            # Get all folders with videos in the directory at path
            video_folders = [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]
            for f in video_folders:
                self.num_videos += 1
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
                    n = len([x for x in os.listdir(s) if os.path.isfile(os.path.join(s, x)) and x.endswith(".png")])
                    if n < sequence_length:
                        #print(sequence_length, n)
                        continue

                    # Generate dict_keys of form f(original)_index for all videos
                    # * f(original) is a bijective mapping of original ids to the values 0 to len(dataset)
                    #   The mapping is necessary to later retrieve the samples by an id between 0 and len(dataset),
                    #   otherwise some ids would not have a corresponding sample
                    # * index is the number of samples generated from the same original that have been added before.
                    #   This gives all samples generated from the same original and the original different ids with the same beginning as an identifier
                    if self.generate_coupled == True:
                        if original in mapping_dict:
                            new_key, num_samples = mapping_dict[original]
                            mapping_dict[original] = (new_key, num_samples + 1)
                        else:
                            new_key = counter
                            num_samples = 0
                            mapping_dict[original] = (new_key, 1)
                            counter += 1

                        key = str(new_key) + "_" + str(num_samples)

                    # If not coupled, just map samples to 0..len chronologically by processing order
                    else:
                        key = counter
                        counter += 1
                    self.frame_dir[key] = (s, actor, original, n)
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

        sample_list = []
        label_list = []
        # Not generate_coupled: Just take the one sample (frames from one video) with key idx
        if self.generate_coupled == False:
            sample_list.append(self.frame_dir[idx])

        # If generate_coupled: Take all samples starting with idx_
        else:
            i = 0
            while ((str(idx) + "_" + str(i))) in self.frame_dir:
                sample_list.append(self.frame_dir[str(idx) + "_" + str(i)])
                i += 1

        samples = []
        # Load all selected samples
        for sample in sample_list:
            this_sample = []
            whole_path, actor, original, n = sample
            label_list.append(actor == -1)  # Append label 1 if actor==-1 (so if video not fake)

            # get all images at whole_path/
            image_names = [f for f in os.listdir(whole_path) if
                           os.path.isfile(os.path.join(whole_path, f)) and f.endswith(".png")]
            image_names = image_names[0:10]

            # Read all images into an image list and stack to 1 numpy matrix
            for name in image_names:
                this_sample.append(io.imread(whole_path + "/" + name))
            image_matrix = np.stack(this_sample)
            samples.append(image_matrix)
        samples = np.stack(samples)
        sample = {"sequences": samples, "labels": np.stack(label_list)}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        samples, labels = sample["sequences"], sample["labels"]

        # swap color axis because
        # numpy image: num_frames x H x W x C
        # torch image: num_frames x C X H X W
        samples = torch.from_numpy(samples.transpose((0, 1, 4, 2, 3)))
        labels = torch.tensor(labels[0])
        return {"image": samples.float(),
                "label"   : torch.tensor(labels).long()}


def my_collate(batch):
    data = np.concatenate([b["sequences"] for b in batch], axis=0)
    targets = [b["labels"] for b in batch]
    sample = {"sequences": data, "labels": targets}
    return sample


if __name__ == '__main__':
    # test /example
    d = [
            "/home/anna/Desktop/Uni/WiSe19/DL4CV/data/FaceForensics/set1/manipulated_sequences/Deepfakes/c40/sequences_128x128_skip_5_uniform",
            "/home/anna/Desktop/Uni/WiSe19/DL4CV/data/FaceForensics/set1/original_sequences/youtube/c40/sequences_128x128_skip_5_uniform"]
    test_dataset = FaceForensicsVideosDataset(d, generate_coupled=False, sequence_length=10, transform=ToTensor())
    print(test_dataset.__len__())
    train_list, val_list = test_dataset.get_train_val_lists(0.9, 0.01)
    print(len(train_list), len(val_list))
    dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=4, shuffle=True,
                                                 collate_fn=my_collate,  # use custom collate function here
                                                 pin_memory=True)

    for i, sample in enumerate(dataset_loader):
        # print("->", sample["sequences"].shape)
        # print(sample["labels"])
        pass
