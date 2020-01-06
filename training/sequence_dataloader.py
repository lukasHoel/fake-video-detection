from skimage import io
from torch.utils import data
import torch
import os
import numpy as np
import math
from tqdm.auto import tqdm


class FaceForensicsVideosDataset(data.Dataset):
    def __init__(self, directories, num_frames, generate_coupled=False, transform=None, max_number_videos_per_directory=-1):
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
        self.num_frames=num_frames
        self.max_number_videos_per_directory = max_number_videos_per_directory

        self.frame_dir = {}
        counter = 0
        for path in directories:
            # Get all folders with videos in the directory at path
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
                for s in sequence_folders:
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

                    #this_sample = []

                    # Read all images into an image list and stack to 1 numpy matrix
                    image_paths = []
                    for name in image_names:
                        image_paths.append(s + "/" + name)
                        #this_sample.append(io.imread(s + "/" + name))
                    #image_matrix = np.stack(this_sample)

                    key = counter
                    counter += 1
                    self.frame_dir[key] = (False, image_paths, label)
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

        is_loaded, sequence, label = self.frame_dir[idx]

        if not is_loaded:
            print("Sequence {} not loaded, will load now".format(idx))
            this_sample = []
            for name in tqdm(sequence):
                this_sample.append(io.imread(name))
            sequence = np.stack(this_sample)
            self.frame_dir[idx] = (True, sequence, label)
            print("Loaded sequence {} into global dataset directory".format(idx))

        samples = np.stack([sequence])
        #print(samples.shape)
        sample = {"image": samples, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        samples, labels = sample["image"], sample["label"]

        # swap color axis because
        # numpy image: num_frames x H x W x C
        # torch image: num_frames x C X H X W
        samples = torch.from_numpy(samples.transpose((0, 1, 4, 2, 3)))
        labels = torch.tensor(labels)
        return {"image": samples.float(),
                "label"   : torch.tensor(labels).long()}


def my_collate(batch):
    data = np.concatenate([b["image"] for b in batch], axis=0)
    targets = [b["label"] for b in batch]
    sample = {"image": data, "label": targets}
    return sample


if __name__ == '__main__':
    # test /example
    d = ["/home/anna/Desktop/Uni/WiSe19/DL4CV/adl4cv/data/FaceForensics/manipulated_sequences/Deepfakes/c40/sequences_299x299_5seq@10frames_skip_5_uniform"]
    test_dataset = FaceForensicsVideosDataset(d, generate_coupled=False, num_frames=5, transform=ToTensor())
    print(test_dataset.__len__())
    train_list, val_list = test_dataset.get_train_val_lists(0.9, 0.01)
    print(len(train_list), len(val_list))
    dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=4, shuffle=True,
                                                 collate_fn=my_collate,  # use custom collate function here
                                                 pin_memory=True)

    for i, sample in enumerate(dataset_loader):
        print("->", sample["image"].shape)
        print(sample["label"])
        pass

