import numpy as np
import sys
import cv2

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
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
    img1 = cv2.imread(sys.argv[1])
    img2 = cv2.imread(sys.argv[2])

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
    cv2.waitKey()

    flow = cv2.calcOpticalFlowFarneback(img1,img2, None, 0.5, 3, 10, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    cv2.imshow("flow HSV", draw_hsv(flow))
    cv2.waitKey()

    res1 = warp_flow(img1, flow)
    res2 = warp_flow(img2, flow)

    cv2.imshow("warp-img1", res1)
    cv2.imshow("warp-img2", res2)
    cv2.waitKey()

    cv2.imwrite("warped_farneback.png", res1)
