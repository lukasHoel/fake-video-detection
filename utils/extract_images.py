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
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
}
COMPRESSION = ['c0', 'c23', 'c40']

def extract_from_directory(data_path, dataset, compression,
                           num_sequences=5, frames_per_sequence=10, skip_frames=5,
                           size=128, padding=30,
                           sample_mode='uniform',
                           num_videos='all'):
    """
    Extracts sequences from all videos in <data_path>/<dataset>/<compression>/videos into <data_path>/<dataset>/<compression>/sequences.
    Will use the FaceForensics file structure to identify all videos of dataset type specified.
    A sequence is a number of face-cropped images starting from a random/uniform frame number inside of the video with configurable number of
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
    :param padding: how much padding shall be used around detected face crop in each direction (to capture all of the face). Will only be used by the haarcascades classifier default: 30
    :param sample_mode: whether samples shall be selected uniform or random. default: uniform.
                        Uniform means deterministic selection of frame-numbers and random means random selection of frame-numbers
                        for the first frame in a sequence.
    :param num_videos: How many videos to extract into sequences of images. Default: all

    :return:
    """
    # Prepare folder names and paths
    videos_path = join(data_path, DATASET_PATHS[dataset], compression, 'videos')
    sequence_suffix = '_' + str(size) + 'x' + str(size) + '_' + str(num_sequences) + 'seq@' + str(frames_per_sequence) + 'frames_skip_' + str(skip_frames) + '_' + sample_mode
    sequences_path = join(data_path, DATASET_PATHS[dataset], compression, 'sequences' + sequence_suffix)

    # Only iterate over num_videos
    videos = os.listdir(videos_path)
    if num_videos != 'all':
        videos = videos[:int(num_videos)]

    # Extract sequences for every video
    for video in tqdm(videos):
        video_name = video.split('.')[0] # e.g. 000_003
        mask_path = join(data_path, DATASET_PATHS[dataset], 'masks', 'videos', video) # path to the mask for this video
        sequences = extract_from_video(join(videos_path, video),
                                       int(num_sequences), int(frames_per_sequence), int(skip_frames),
                                       int(size), int(padding),
                                       sample_mode,
                                       mask_path)
        save_sequences(sequences, join(sequences_path, video_name))


def extract_from_video(video_path, num_sequences, frames_per_sequence, skip_frames, size, padding, sample_mode, mask_path):
    print("Extract from video {}".format(video_path))

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    required_frames_per_sequence = (frames_per_sequence - 1) * skip_frames

    sequences = [[None for x in range(frames_per_sequence)] for y in range(num_sequences)]

    for seq_nr in range(num_sequences):
        if sample_mode == 'uniform':
            available_frames_per_sequence = total_frames / num_sequences # uniformly divide video into equally large frame blocks
            if available_frames_per_sequence < required_frames_per_sequence:
                # if there are not enough frames to not overlap between sequences, then we allow overlapping by that factor
                first_frame_number = seq_nr * (required_frames_per_sequence/available_frames_per_sequence) * 2
            else:
                # if there are enough frames, we do not overlap between sequences and start each sequence at a new frame.
                first_frame_number = seq_nr * available_frames_per_sequence
        else:
            first_frame_number = random.randrange(total_frames - required_frames_per_sequence)

        first_frame_number = int(first_frame_number)
        print("Extract sequence {} from frame {}".format(seq_nr, first_frame_number))
        num_crop_fails = 0
        num_mask_crop_fails = 0

        for frame_nr in range(frames_per_sequence):
            video_pos = int(first_frame_number + frame_nr*skip_frames) # calculate next video frame position
            video.set(cv2.CAP_PROP_POS_FRAMES, video_pos) # set next video frame position
            ret, img = video.read() # read next image

            if ret: # if image could be extracted from video - "should always work"
                found_crop, cropped_img = crop_face_from_mask(img, size, mask_path, video_pos)
                if found_crop: # only then save the image ... else try cropping with haarcascade classifier
                    sequences[seq_nr][frame_nr] = cropped_img
                else:
                    num_mask_crop_fails += 1
                    found_crop, cropped_img = crop_face_from_image(img, size, padding) # crop image on face
                    if found_crop: # only then save the image ... else we have one less image
                        sequences[seq_nr][frame_nr] = cropped_img
                    else:
                        num_crop_fails += 1

        print("Finished extracting sequence {} with {} unfound face-crops and {} mask_crop_fails".format(seq_nr, num_crop_fails, num_mask_crop_fails))

    video.release()
    print("Finished extracting from video {}".format(video_path))

    return sequences

def crop_face_from_mask(face_img, size, mask_path, frame_number):
    if mask_path == None:
        return (False, None)

    # load mask video and extract image from required frame_number
    mask_video = cv2.VideoCapture(mask_path)
    mask_video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, mask_img = mask_video.read()  # read next image
    mask_video.release()
    if not ret:
        return (False, None)

    # extract the location of the mask from the mask image (via opencv contours function)
    img_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 70, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours == None or len(contours) == 0:
        return (False, None)

    c = max(contours, key=cv2.contourArea) # find the biggest area and use as face crop
    x, y, w, h = cv2.boundingRect(c)

    # visualize the location of the mask in the mask image
    #img = np.copy(mask_img)
    #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #cv2.imshow("Extraction of mask from mask_img", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # calculate where to crop in face_img which could be of different resolution than mask_img
    mask_h, mask_w, _ = mask_img.shape
    face_h, face_w, _ = face_img.shape
    scale_w = 1.0 * face_w / mask_w
    scale_h = 1.0 * face_h / mask_h
    x = int(x * scale_w)
    w = int(w * scale_w)
    y = int(y * scale_h)
    h = int(h * scale_h)

    # visualize the location of the mask in the face image
    #img = np.copy(face_img)
    #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #cv2.imshow("Extraction of mask from face_img", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # crop and resize the face_img
    # this creates a square image by taking the largest of w/h as length of the square-crop
    nx, ny, nr = calc_square_bounding_box(x, y, w, h, 0) # always zero padding because mask is precise
    res = crop_and_resize_image(face_img, size, nx, ny, nr, nr)

    # this creates a square image by resizing (linear interpolation!) the image to the square resolution
    # this might be unpreferred because it might blur the artifacts in fake images due to interpolation
    #res = crop_and_resize_image(face_img, size, x, y, w, h)

    # visualize the result crop
    #cv2.imshow("Extraction of mask from face_img after crop and resize", res[1])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return res

def crop_and_resize_image(image, size, x, y, w, h):
    crop = image[y:y+h, x:x+w]
    if isinstance(crop, np.ndarray) and len(crop) > 0 and len(crop[0]) > 0:
        # check necessary because we this might be None/empty sometimes... why?
        crop_resize = cv2.resize(crop, (size, size))
        return (True, crop_resize)
    else:
        return (False, None)

def calc_square_bounding_box(x, y, w, h, padding):
    r = max(w, h) / 2  # get radius so that we can extract a square image
    centerx = x + w / 2
    centery = y + h / 2
    nx = int(centerx - r - (padding / 2))
    ny = int(centery - r - (padding / 2))
    nr = int(r * 2 + padding)

    return (nx, ny, nr)


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
        minSize=(80, 80)
    )

    if(len(faces) == 0):
        return (False, image)
    else:
        (x, y, w, h) = faces[0]

        nx, ny, nr = calc_square_bounding_box(x, y, w, h, padding)

        # Use this to show the detected face (green: detected + padding --> this is what will be extracted), (blue: original detected)
        #cv2.rectangle(image, (nx, ny), (nx + nr, ny + nr), (0, 255, 0), 2)
        #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        #plt.imshow(image)
        #plt.show()

        return crop_and_resize_image(image, size, nx, ny, nr, nr)

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
    p.add_argument('--sample_mode', type=str,
                   choices=['uniform','random'],
                   default='uniform')
    p.add_argument('--num_videos', type=str,
                   default='all')
    args = p.parse_args()

    if args.dataset == 'all':
        for dataset in DATASET_PATHS.keys():
            args.dataset = dataset
            extract_from_directory(**vars(args))
    else:
        extract_from_directory(**vars(args))