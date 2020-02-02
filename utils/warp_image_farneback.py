'''
Author: Lukas Hoellein

Calculates optical-flow and warped images via OpenCV Farneback method.
'''

import numpy as np
import sys
import cv2

def call_farneback_of(img1, img2):
    """
    Calls OpenCV's farneback Optical Flow Method. 
    Uses different parameters than the function doing the same for later warping.
    Only necessary to change Farneback Parameters for Optical Flow functions here.
    """
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 6, 5, 4, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

def calc_flow(img1, img2):
    """Returns standard Optical Flow (x, y) from img1 to img2."""
    flow = call_farneback_of(img1, img2)
    return flow
    
def calc_flow_mag_ang(img1, img2):
    """Returns the Optical Flow from img1 to img2 as the (angle, magnitude) representation."""
    flow = call_farneback_of(img1, img2)
    flowma = np.zeros((img.shape[0], img.shape[1], 2))
    
    # Magnitude
    flowma[:,:,0] = np.sqrt(flow[:,:,0] * flow[:,:,0] + flow[:,:,1] * flow[:,:,1])
    
    # Angle
    nullen = flow[:,:,0] == 0
    werte = flow[:,:,0] != 0
    # Set all elements where flow[:,:,0] is not 0 to calculated angle
    flowma[:,:,layer][werte] =  np.arctan(flow[:,:,1][werte] / flow[:,:,0][werte])
    # Set all elements were flow[:,:,0] was 0 to 0 (so no division by 0)
    flowma[:,:,layer][nullen] = 3.1425269 / 2
    return flowma

def calc_flow3D(img1, img2, components=["x", "y", "angle"]):
    """
    Returns the Optical Flow as a (w x h x 3) tensor using the representations given in components.
    Possibilities for components: x, y, angle, magnitude
    """
    flow3D = np.zeros(img1.shape)
    flow = call_farneback_of(img1, img2)
    
    layer = 0
    if "x" in components:
        flow3D[:,:,layer] = flow[:,:,0]
        layer += 1

    if "y" in components:
        flow3D[:,:,layer] = flow[:,:,1]
        layer += 1

    if "magnitude" in components:
        flow3D[:,:,layer] = np.sqrt(flow[:,:,0] * flow[:,:,0] + flow[:,:,1] * flow[:,:,1])
        layer += 1

    if "angle" in components:
        nullen = flow[:,:,0] == 0
        werte = flow[:,:,0] != 0
        # Set all elements where flow[:,:,0] is not 0 to calculated angle
        flow3D[:,:,layer][werte] =  np.arctan(flow[:,:,1][werte] / flow[:,:,0][werte])
        # Set all elements were flow[:,:,0] was 0 to 0 (so no division by 0)
        flow3D[:,:,layer][nullen] = 3.1425269 / 2
    
    # Remove all remaining nans wherever they come from
    flow3D[np.isnan(flow3D)] = 0
    return flow3D

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
