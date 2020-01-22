'''
Author: Lukas Hoellein

Calculates optical-flow and warped images via OpenCV Farneback method.
'''

import numpy as np
import sys
import cv2

def warp_from_images(img1, img2):
    flow = calc_flow(img1, img2)
    warp = warp_flow(img2, flow)
    return warp

def calc_flow(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 4, 5, 4, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    return flow

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    #flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

if __name__ == '__main__':
    #img1 = cv2.imread(sys.argv[1])
    #img2 = cv2.imread(sys.argv[2])
    import matplotlib.pyplot as plt

    # SETUP sample images
    img1 = np.zeros((1000, 1000, 3)).astype(np.float32)
    img2 = np.zeros_like(img1)
    img1[100:400, 200:500, :] = 255
    img2[101:401, 201:501, :] = 255

    # SHOW sample images
    plt.imshow(img1)
    plt.show()
    plt.imshow(img2)
    plt.show()

    # CALCULATE optical flow + show it
    flow = calc_flow(img1, img2)
    plt.imshow(draw_hsv(flow))
    plt.show()

    # CALCULATE warped image + show it
    res = warp_flow(img2, flow)
    plt.imshow(res)
    plt.show()
