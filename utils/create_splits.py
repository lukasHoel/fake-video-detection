'''
Author: Lukas Hoellein

Splits a sequence_folder created by the "extract_images" script into train/val/test set according to the official
FaceForensics train/val/test split found in https://github.com/ondyari/FaceForensics/tree/master/dataset/splits
'''

import argparse
import os
import shutil
import random
import json
from os.path import join

COMPRESSION = ['c0', 'c23', 'c40']

DATASET_PATHS = {
    #'original1': 'original_sequences/actors',
	'original2': 'original_sequences/youtube',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
}

def parse_splits(path_to_splits):
    with open(join(path_to_splits, "train.json")) as train_split:
        train_split = [item for sublist in json.load(train_split) for item in sublist]

    with open(join(path_to_splits, "val.json")) as val_split:
        val_split = [item for sublist in json.load(val_split) for item in sublist]

    with open(join(path_to_splits, "test.json")) as test_split:
        test_split = [item for sublist in json.load(test_split) for item in sublist]

    return (train_split, val_split, test_split)

def create_splits_from_file(input, sequence, compression, split_dir):
    '''
        Splits a sequence_folder created by the "extract_images" script into train/val/test set according to the official
        FaceForensics train/val/test split found in https://github.com/ondyari/FaceForensics/tree/master/dataset/splits
    '''

    train_split, val_split, test_split = parse_splits(split_dir)

    for dataset in DATASET_PATHS.values():
        absolute_path_to_sequence = join(input, dataset, compression, sequence)
        absolute_train_path = join(absolute_path_to_sequence, "train")
        absolute_val_path = join(absolute_path_to_sequence, "val")
        absolute_test_path = join(absolute_path_to_sequence, "test")
        os.makedirs(absolute_train_path, exist_ok=True)
        os.makedirs(absolute_val_path, exist_ok=True)
        os.makedirs(absolute_test_path, exist_ok=True)

        train_counter = 0
        val_counter = 0
        test_counter = 0

        for seq in os.listdir(absolute_path_to_sequence):
            for seq_path in seq.split("_"):
                path = join(absolute_path_to_sequence, seq)
                if seq_path in train_split:
                    #print("Would move {} to {}".format(path, absolute_train_path))
                    shutil.move(path, absolute_train_path)
                    train_counter += 1
                    break
                elif seq_path in val_split:
                    #print("Would move {} to {}".format(path, absolute_val_path))
                    shutil.move(path, absolute_val_path)
                    val_counter += 1
                    break
                elif seq_path in test_split:
                    #print("Would move {} to {}".format(path, absolute_test_path))
                    shutil.move(path, absolute_test_path)
                    test_counter += 1
                    break
                elif seq_path in ["train", "val", "test"]:
                    print("Ignoring {}".format(seq_path))
                    pass
                else:
                    raise Exception("{} not found in any split".format(sequence))

        print("Train: {}, Val: {}, Test: {}".format(train_counter, val_counter, test_counter))

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--input', '-i', type=str)
    p.add_argument('--sequence', '-s', type=str)
    p.add_argument('--compression', '-c', type=str, choices=COMPRESSION,
                   default='c40')
    p.add_argument('--split_dir', '-sd', type=str)

    args = p.parse_args()

    create_splits_from_file(**vars(args))