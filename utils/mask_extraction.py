import argparse
import cv2
import os
import random
from os.path import join
from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DATASETS = ['FaceSwap', 'Deepfakes', 'Face2Face', 'NeuralTextures']

if __name__ == '__main__':
    for dataset in DATASETS:
        video = cv2.VideoCapture("C:\\Users\\admin\\Desktop\\FaceForensics\\manipulated_sequences\\" + dataset + "\\masks\\videos\\033_097.mp4")
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, img = video.read()
        thresh = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(thresh, 70, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # find the biggest area
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Extraction of mask from {}".format(dataset), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()