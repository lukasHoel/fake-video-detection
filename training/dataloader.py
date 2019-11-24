from skimage import io
from torch.utils import data
import torch
import os
import numpy as np


class FaceForensicsVideosDataset(data.Dataset):
    def __init__(self, directories, generate_coupled=False, transform=None):
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
                sequence_folders = [os.path.join(path, f, x) for x in  os.listdir(os.path.join(path, f)) if not os.path.isfile(os.path.join(path, f, x))]
                for s in sequence_folders:
                    # Discard empty sample folders
                    n = len([x for x in os.listdir(s) if os.path.isfile(os.path.join(s, x)) and x.endswith(".png")])
                    if n == 0:
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
            label_list.append(actor == -1)      # Append label 1 if actor==-1 (so if video not fake)

            #get all images at whole_path/
            image_names = [f for f in os.listdir(whole_path) if
                     os.path.isfile(os.path.join(whole_path, f)) and f.endswith(".png")]

            # Read all images into an image list and stack to 1 numpy matrix
            for name in image_names:
                this_sample.append(io.imread(whole_path + "/" + name))
            image_matrix = np.stack(this_sample)
            samples.append(image_matrix)

        sample = {'samples': samples, 'labels': label_list, 'original_id': original}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        samples, labels, vid_name = sample['samples'], sample['labels'], sample['original_id']

        # swap color axis because
        # numpy image: num_frames x H x W x C
        # torch image: num_frames x C X H X W
        samples = [torch.from_numpy(images.transpose((0, 3, 1, 2))) for images in samples]
        return {'images'    : samples,
                'labels'     : labels,
                'original_id': vid_name}


# test /example
d = ["C:/Users/admin/Desktop/FaceForensics/manipulated_sequences/Face2Face/c40/sequences"]
test_dataset = FaceForensicsVideosDataset(d, generate_coupled=True, transform=ToTensor())
dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)
sample = test_dataset.__getitem__(24)
print(test_dataset.__len__())
print("Number of samples: ", len(sample["images"]))
print("Original id: ", sample["original_id"])  # id of original video
print("Labels: ", sample["labels"])
print("Tensor shape of one sample in the list: ", sample["images"][0].shape)
