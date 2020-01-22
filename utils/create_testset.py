'''
Author: Lukas Hoellein

Assumes equal amount of videos in all dataset directories!!
    Will move for all datasets including original (only selecting a few datasets is not possible due to ambiguities!)

    For all datasets: Recursively moves a uniform/random sampled subset of directories in 'input/dataset/compression/sequence' to
    'output/FaceForensics_Testset/dataset/compression/sequence'. The number of directories to be moved can be controlled
    by the percentage parameter.
'''

import argparse
import os
import shutil
import random
from os.path import join

COMPRESSION = ['c0', 'c23', 'c40']
SAMPLE_MODE = ['random', 'uniform']

DATASET_PATHS = {
    #'original1': 'original_sequences/actors',
	'original2': 'original_sequences/youtube',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
}

def extract_testset_from_directory(input, output, sequence, compression, percentage, sample_mode):
    '''
    Assumes equal amount of videos in all dataset directories!!
    Will move for all datasets including original (only selecting a few datasets is not possible due to ambiguities!)

    For all datasets: Recursively moves a uniform/random sampled subset of directories in 'input/dataset/compression/sequence' to
    'output/FaceForensics_Testset/dataset/compression/sequence'. The number of directories to be moved can be controlled
    by the percentage parameter.

    :param input: root directory of the FaceForensics folder.
    :param output: root directory where the output folder "FaceForensics_Testset" shall be created
    :param sequence: from which sequence to extract into the testset (e.g. from sequences_299x299_skip_5_uniform)
    :param compression: from which compression level to extract into the testset (e.g. from c40)
    :param percentage: how many sequences to extract into the testset (e.g. 10%)
    :param sample_mode: how to extract (e.g. uniform: every i-th, random)
    :return:
    '''

    relative_path_to_sequence = join(DATASET_PATHS['Deepfakes'], compression, sequence)
    absolute_path_to_sequence = join(input, relative_path_to_sequence)
    number_subdirs = sum(os.path.isdir(absolute_path_to_sequence + '\\' + i) for i in os.listdir(absolute_path_to_sequence))
    percentage = float('0.' + percentage)
    testset_length = int(number_subdirs * percentage)
    testset_sequences = []
    #print("Number of directories in {}: {}".format(absolute_path_to_sequence, number_subdirs))
    #print("Testset length: {}".format(testset_length))

    if number_subdirs < 1:
        return

    if sample_mode == 'uniform':
        skip_rate = int(number_subdirs / testset_length)
        #print("Skip rate: {}".format(skip_rate))
        i = 0
        for seq in os.listdir(absolute_path_to_sequence):
            if i%skip_rate == 0:
                #print("Append sequence: {}".format(absolute_path_to_sequence + '\\' + sequence))
                testset_sequences.append(seq)
            i += 1
    else:
        indices = random.sample(range(number_subdirs), testset_length)
        #print("Select these indices: {}".format(indices))
        i = 0
        for seq in os.listdir(absolute_path_to_sequence):
            if i in indices:
                #print("Append sequence: {}".format(absolute_path_to_sequence + '\\' + sequence))
                testset_sequences.append(seq)
            i += 1

    testset_sequences = check_symmetric_sequences(absolute_path_to_sequence, testset_sequences, percentage, number_subdirs)
    orig_testset_sequences = extract_original_sequences(testset_sequences)

    for dataset in DATASET_PATHS.keys():
        # construct prefix for every dataset
        relative_dataset_path = join(DATASET_PATHS[dataset], compression, sequence)
        prefix = join(input, relative_dataset_path)

        # append prefix to testset sequences
        if dataset != 'original2':
            testset_sequences_with_prefix = [join(prefix, s) for s in testset_sequences]
        else:
            testset_sequences_with_prefix = [join(prefix, s) for s in orig_testset_sequences]

        # extract_sequences(outdir, sequences_with_prefix)
        outdir = join(output, 'FaceForensics_Testset', relative_dataset_path)
        extract_sequences(outdir, testset_sequences_with_prefix)


def check_symmetric_sequences(sequence_path, testset_sequences, percentage, total_length):
    testset_sequences_with_symmetric = []
    for s in testset_sequences:
        testset_sequences_with_symmetric.append(s)
        (first, second) = s.split('_')
        symmetric_sequence = second + '_' + first

        print("Checking {} for symmetric sequence {}".format(s, symmetric_sequence))
        for seq in os.listdir(sequence_path):
            if seq == symmetric_sequence:
                print("Found symmetric sequence {}".format(seq))
                testset_sequences_with_symmetric.append(seq)

    new_length = len(testset_sequences_with_symmetric)
    new_percentage = new_length / total_length
    #print("New length with symmetric sequences: {} New percentage: {}".format(new_length, new_percentage))

    if new_percentage > (percentage + 0.02):
        # epsilon of 2% allowed
        allowed_length = int(percentage * total_length)

        (last1, last2) = testset_sequences_with_symmetric[allowed_length-1].split('_')
        (next1, next2) = (last1, last2) = testset_sequences_with_symmetric[allowed_length].split('_')

        if last1 == next2 and last2 == next1:
            allowed_length = allowed_length + 1

        testset_sequences_with_symmetric = testset_sequences_with_symmetric[:allowed_length]
        #print("Reduced symmetric testset from {} to {} elements (allowed are: {})".format(new_length, len(testset_sequences_with_symmetric), allowed_length))

        return testset_sequences_with_symmetric

def extract_original_sequences(testset_sequences):
    orig_testset_sequences = []
    for s in testset_sequences:
        orig_testset_sequences.append(s.split('_')[0])
    return set(orig_testset_sequences) # can contain duplicate values

def extract_sequences(output, testset_sequences):
    #print("Create {}".format(output))
    os.makedirs(output, exist_ok=True)
    for sequence in testset_sequences:
        #print("Would now move {} to {}".format(sequence, output))
        shutil.move(sequence, output)

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--input', '-i', type=str)
    p.add_argument('--output', '-o', type=str)
    p.add_argument('--sequence', '-s', type=str)
    p.add_argument('--compression', '-c', type=str, choices=COMPRESSION,
                   default='c0')
    p.add_argument('--percentage', '-p', type=str,
                   default='10')
    p.add_argument('--sample_mode', '-m', type=str, choices=SAMPLE_MODE,
                   default='uniform')

    args = p.parse_args()

    extract_testset_from_directory(**vars(args))