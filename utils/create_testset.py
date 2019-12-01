import argparse
import os
import shutil
import random
from os.path import join

COMPRESSION = ['c0', 'c23', 'c40']
SAMPLE_MODE = ['random', 'uniform']

DATASET_PATHS = {
    'original1': 'original_sequences/actors',
	'original2': 'original_sequences/youtube',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    #'NeuralTextures': 'manipulated_sequences/NeuralTextures'
}

def extract_testset_from_directory(input, output, sequence, dataset, compression, percentage, sample_mode):
    '''
    Recursively moves a uniform/random sampled subset of directories in 'input/dataset/compression/sequence' to
    'output/FaceForensics_Testset/dataset/compression/sequence'. The number of directories to be moved can be controlled
    by the percentage parameter.

    :param input: root directory of the FaceForensics folder.
    :param output: root directory where the output folder "FaceForensics_Testset" shall be created
    :param sequence: from which sequence to extract into the testset (e.g. from sequences_299x299_skip_5_uniform)
    :param dataset: from which dataset to extract into the testset (e.g. from Deepfakes)
    :param compression: from which compression level to extract into the testset (e.g. from c40)
    :param percentage: how many sequences to extract into the testset (e.g. 10%)
    :param sample_mode: how to extract (e.g. uniform: every i-th, random)
    :return:
    '''
    path_to_sequence = join(DATASET_PATHS[dataset], compression, sequence)
    sequence_path = join(input, path_to_sequence)
    number_subdirs = sum(os.path.isdir(sequence_path + '\\' + i) for i in os.listdir(sequence_path))
    percentage = float('0.' + percentage)
    testset_length = int(number_subdirs * percentage)
    testset_sequences = []
    #print("Number of directories in {}: {}".format(sequence_path, number_subdirs))
    #print("Testset length: {}".format(testset_length))

    if sample_mode == 'uniform':
        skip_rate = int(number_subdirs / testset_length)
        #print("Skip rate: {}".format(skip_rate))
        i = 0
        for sequence in os.listdir(sequence_path):
            if i%skip_rate == 0:
                #print("Append sequence: {}".format(sequence_path + '\\' + sequence))
                testset_sequences.append(sequence_path + '\\' + sequence)
            i += 1
    else:
        indices = random.sample(range(number_subdirs), testset_length)
        #print("Select these indices: {}".format(indices))
        i = 0
        for sequence in os.listdir(sequence_path):
            if i in indices:
                #print("Append sequence: {}".format(sequence_path + '\\' + sequence))
                testset_sequences.append(sequence_path + '\\' + sequence)
            i += 1

    outdir = output + '\\' + 'FaceForensics_Testset' + '\\' + path_to_sequence
    extract_sequences(outdir, testset_sequences)

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
    p.add_argument('--dataset', '-d', type=str,
                   choices=list(DATASET_PATHS.keys()) + ['all'],
                   default='all')
    p.add_argument('--compression', '-c', type=str, choices=COMPRESSION,
                   default='c0')
    p.add_argument('--percentage', '-p', type=str,
                   default='10')
    p.add_argument('--sample_mode', '-m', type=str, choices=SAMPLE_MODE,
                   default='uniform')

    args = p.parse_args()

    if args.dataset == 'all':
        for dataset in DATASET_PATHS.keys():
            args.dataset = dataset
            extract_testset_from_directory(**vars(args))
    else:
        extract_testset_from_directory(**vars(args))