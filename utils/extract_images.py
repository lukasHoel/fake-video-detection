import argparse
import cv2
import os
import random
from os.path import join
from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DATASET_PATHS = {
    'original1': 'original_sequences/actors',
	'original2': 'original_sequences/youtube',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap'
}
COMPRESSION = ['c0', 'c23', 'c40']

def extract_from_directory(data_path, dataset, compression,
                           num_sequences=5, frames_per_sequence=10, skip_frames=5,
                           size=128, padding=30):
    """
    Extracts sequences from all videos in <data_path>/<dataset>/<compression>/videos into <data_path>/<dataset>/<compression>/sequences.
    Will use the FaceForensics file structure to identify all videos of dataset type specified.
    A sequence is a number of face-cropped images starting from a random frame number inside of the video with configurable number of
    frames to skip between two images.
    Videos will be saved in a subdirectory structure as follows:
    <root_of_dataset>
        <root_of_dataset>/videos
            <here lie all videos to extract from, e.g. foo.mp4, bar.mp4>
        <root_of_dataset>/sequences
            <foo>: subdirectory with the name of the original video (without datatype suffix e.g. without .mp4)
                <0>: number of sequence
                    0000.png: first picture in this sequence
                    0001.png: second picture in this sequence
                <1>
                    0000.png
                    0001.png
            <bar>
                <0>
                    0000.png
                    0001.png
                <1>
                    0000.png
                    0001.png

    :param data_path: path of FaceForensics root directory
    :param dataset: of which dataset to choose from. See DATASET_PATHS
    :param compression: of which compression level to choose from. See COMPRESSION
    :param num_sequences: how many sequences to extract. default: 5
    :param frames_per_sequence: how many frames each sequence shall contain. default: 10
    :param skip_frames: how many frames shall be skipped between two frames (to capture changes in expression) default: 5
    :param size: how big each frame shall be. Is considered to give both width and height, thus resulting image is quadratic. default: 128
    :param padding: how much padding shall be used around detected face crop in each direction (to capture all of the face) default: 30

    :return:
    """
    videos_path = join(data_path, DATASET_PATHS[dataset], compression, 'videos')
    sequences_path = join(data_path, DATASET_PATHS[dataset], compression, 'sequences')
    for video in tqdm(os.listdir(videos_path)):
        sequence_folder = video.split('.')[0] # name like the video is called
        sequences = extract_from_video(join(videos_path, video),
                                       int(num_sequences), int(frames_per_sequence), int(skip_frames),
                                       int(size), int(padding))
        save_sequences(sequences, join(sequences_path, sequence_folder))


def extract_from_video(video_path, num_sequences, frames_per_sequence, skip_frames, size, padding):
    print("Extract from video {}".format(video_path))

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    sequences = [[None for x in range(frames_per_sequence)] for y in range(num_sequences)]

    for seq_nr in range(num_sequences):
        first_frame_number = random.randrange(total_frames)
        print("Extract sequence {} from frame {}".format(seq_nr, first_frame_number))
        num_crop_fails = 0

        for frame_nr in range(frames_per_sequence):
            video_pos = first_frame_number + frame_nr*skip_frames # calculate next video frame position
            video.set(cv2.CAP_PROP_POS_FRAMES, video_pos) # set next video frame position
            ret, img = video.read() # read next image
            found_crop, img = crop_face_from_image(img, size, padding) # crop image on face

            if found_crop: # only then save the image ... else we have one less image
                sequences[seq_nr][frame_nr] = img
            else:
                num_crop_fails += 1

        print("Finished extracting sequence {} with {} unfound face-crops".format(seq_nr, num_crop_fails))

    video.release()
    print("Finished extracting from video {}".format(video_path))

    return sequences

def crop_face_from_image(image, size, padding):
    """
    Crop face from image using OpenCv "haarcascades" face detector, see: https://www.digitalocean.com/community/tutorials/how-to-detect-and-extract-faces-from-an-image-with-opencv-and-python

    :param image:
    :param size:
    :param padding:
    :return:
    """
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(size - 10, size - 10)
    )

    if(len(faces) == 0):
        return (False, image)
    else:
        (x, y, w, h) = faces[0]

        r = max(w, h) / 2 # get radius so that we can extract a square image
        centerx = x + w / 2
        centery = y + h / 2
        nx = int(centerx - r - padding/2)
        ny = int(centery - r - padding/2)
        nr = int(r * 2 + padding)

        # Use this to show the detected face (green: detected + padding --> this is what will be extracted), (blue: original detected)
        #cv2.rectangle(image, (nx, ny), (nx + nr, ny + nr), (0, 255, 0), 2)
        #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        #plt.imshow(image)
        #plt.show()

        faceimg = image[ny:ny + nr, nx:nx + nr]
        cropped_image = cv2.resize(faceimg, (size, size))

        return (True, cropped_image)

def save_sequences(sequences, path):
    os.makedirs(path, exist_ok=True) # create path ".../sequences/<video_name>"
    for seq_nr in range(len(sequences)):
        sequence_path = join(path, str(seq_nr)) # create path ".../sequences/<video_name>/<seq_nr>" e.g. seq_nr = 0
        os.makedirs(sequence_path, exist_ok=True)
        for img_nr in range(len(sequences[seq_nr])):
            img = sequences[seq_nr][img_nr]
            if isinstance(img, np.ndarray): # check necessary because we might have found no crop in which case the image stayed None
                cv2.imwrite(join(sequence_path, '{:04d}.png'.format(img_nr)), img) # save as e.g. 0000.png, 0101.png, etc.
    return

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--data_path', type=str)
    p.add_argument('--dataset', '-d', type=str,
                   choices=list(DATASET_PATHS.keys()) + ['all'],
                   default='all')
    p.add_argument('--compression', '-c', type=str, choices=COMPRESSION,
                   default='c0')
    p.add_argument('--num_sequences', type=str,
                   default='5')
    p.add_argument('--frames_per_sequence', type=str,
                   default='10')
    p.add_argument('--skip_frames', type=str,
                   default='5')
    p.add_argument('--size', type=str,
                   default='128')
    p.add_argument('--padding', type=str,
                   default='30')
    args = p.parse_args()

    if args.dataset == 'all':
        for dataset in DATASET_PATHS.keys():
            args.dataset = dataset
            extract_from_directory(**vars(args))
    else:
        extract_from_directory(**vars(args))