import math

import cv2 as cv
import constants as const
import numpy as np
from cameraCalibration import getImagesFromVideo, showImage

models = []
lastFrame = []

class Stats(object):
    """
    Welford's algorithm computes variance and mean online
    A class computing the mean of a list each time an element is inserted without having to keep all the items in memory
    """

    def __init__(self):
        self.count, self.M1, self.M2 = 0, 0.0, 0.0

    def add(self, val):
        self.count += 1
        self.delta = val - self.M1
        self.M1 += self.delta / self.count
        self.M2 += self.delta * (val - self.M1)

    @property
    def mean(self):
        return self.M1

    @property
    def variance(self):
        return self.M2 / self.count

    @property
    def std(self):
        return np.sqrt(self.variance)

# Compute for each pixel coordinate and for each channel the mean, varaince and standard deviation
# over a given number of frames of the background video
def backgroundModel(camera, videoType):
    video = cv.VideoCapture(camera + videoType)
    c = int(video.get(cv.CAP_PROP_FRAME_WIDTH ))
    l = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    res = np.empty((l, c, 3), dtype=object)
    frames = getImagesFromVideo(camera, videoType, const.IMAGES_BACKGROUND_NB)
    for i in range(l):
        for j in range(c):
            for k in range(3):
                res[i, j, k] = Stats()

    i = 0
    for frame in frames:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        l,c,_ = frame.shape
        for i in range(l):
            print(str(100*(i+1)/len(frames)) + " %")
            for j in range(c):
                for k in range(3):
                    res[i,j,k].add(frame[i,j,k])
        i +=1
    return res

# Given a background model value and the actual value (of the image with a foreground)
# Computes the number of Standard deviation between the model and the foreground image
def channelDist(model, val, dim):
    delta = model[dim].mean - val[dim]
    if delta < 0:
        delta = -delta
    if model[dim].std > 0.5:
        return delta/model[dim].std
    else :
        return delta * 2

# Compute the weigthed sum of all channel to give the final distance between the model and the image with a foreground
def dist(model, val):
    return const.H_WEIGHT * channelDist(model,val,const.H) + const.S_WEIGHT * channelDist(model,val,const.S) + const.V_WEIGHT * channelDist(model,val,const.V)

# Return black (distance lower than treshold and considered as background)
# or white (distance bigger than treshold and considered as foreground)
def mask(model, val):
    if dist(model,val) > const.THRESHOLD:
        return 255
    else :
        return 0

# When left clicking, display vallues from the channel and total dist
# When right clicking, show the mask
def click_event(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        for k in range(3):
            print("CHANNEL " + str(k) + ": " + str(channelDist(m[y,x], f[y,x], k)))
        print("Total dist : " + str(dist(m[y,x],f[y,x])))
    if event == cv.EVENT_RBUTTONDOWN:
        global showMask
        if showMask :
            showImage(const.WINDOW_NAME, f)
        else :
            showImage(const.WINDOW_NAME, maskF)
        showMask = not showMask

# Return vertical line / use for the kernel of morphology transform in post processing step
def getVerticalLine(size):
    return np.ones(shape=[size, 1], dtype=np.uint8)

# Return horizontal line / use for the kernel of morphology transform in post processing step
def getHorizontalLine(size):
    return np.ones(shape=[1, size], dtype=np.uint8)

# Return an axis aligned cross / use for the kernel of morphology transform in post processing step
def getAxisAlignedCross(size):
    res = np.zeros(shape=size, dtype=np.uint8)
    l = size[0]//2
    c = size[1]//2
    for i in range(size[0]):
        for j in range(size[1]):
            if i == l or j == c:
                res[i,j] = 1
    return res

# Function to compute the mask where white pixel means the pixel is considered as foreground and black mean considered as background.
def substractBackground(camera, videoType, model, frame):
    global lastFrame
    #extracts frame from video
    video = cv.VideoCapture(camera + videoType)
    frameCount = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    c = int(video.get(cv.CAP_PROP_FRAME_WIDTH ))
    l = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    raw = np.empty(shape=[l, c], dtype=np.uint8)

    global m
    global f
    global maskF
    global showMask

    #create the background mask
    # for fc in range(frameCount):
    video.set(cv.CAP_PROP_POS_FRAMES, frame)
    ret, frame = video.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    for i in range(l):
        for j in range(c):
            raw[i,j] = mask(model[i, j], frame[i, j])

    #erode and dilate the background image to fill holes and get rid of as much noise as possible.
    raw = cv.morphologyEx(raw, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    raw = cv.morphologyEx(raw, cv.MORPH_OPEN, getAxisAlignedCross((5,3)))
    raw = cv.morphologyEx(raw, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))

    # Setting global variable for click event (debuging)
    m = model
    f = frame
    maskF = raw
    showMask = True
    #showImage(const.WINDOW_NAME, raw)
    #cv.setMouseCallback(const.WINDOW_NAME, click_event)

    #find the contours of the image
    contours, _ = cv.findContours(raw, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
    big_blobs = []

    #find the big blobs.
    if len(contours) > 0:
        for i in range(len(contours)):
            contour_area = cv.contourArea(contours[i]);
            if contour_area > 5000:
                big_blobs.append(i)

    #only keep the big blobs in order to get rid of noise.
    res = np.zeros(shape=[l, c], dtype=np.uint8)
    for i in range(len(big_blobs)):
        res = cv.drawContours(res, contours, big_blobs[i], 255, cv.FILLED, 8)

    res = cv.bitwise_and(res, raw)
    maskF = res
    # Show keypoints
    #showImage(const.WINDOW_NAME, res, 0)
    lastFrame = res
    cv.imwrite(camera+"foreground.png", res)
    return res

# Gets the foreground mask by subtracting the background from the current frame
def get_foreground_mask(camera, frame):
    res = substractBackground(camera[0], const.VIDEO_TEST, models[camera[2]], frame)
    return res

# Gets the background model for the camera
def get_background_model(camera):
    model = backgroundModel(camera[0], const.VIDEO_BACKGROUND)
    models.append(model)

# Returns the pixels that switched from on to off or viceversa
def get_difference(camera, frame):
    global lastFrame
    lFrame = lastFrame
    #get the background subtraction for this frame
    res = substractBackground(camera[0], const.VIDEO_TEST, models[camera[2]], frame)
    shape = res.shape
    width = shape[1]
    height = shape[0]
    newpixelson = []
    newpixelsoff = []
    #check which pixels changed
    for x in range(height):
        for y in range(width):
            if(res[x,y] != lFrame[x,y]):
                if(res[x,y] != 0):
                    newpixelsoff.append((x,y))
                else:
                    newpixelson.append((x,y))
    
    return newpixelsoff, newpixelson, res

if __name__ == "__main__":
    camArray = [const.CAM1, const.CAM2, const.CAM3, const.CAM4]
    for i in range(4):
        print(str(i))
        model = backgroundModel(camArray[i][0], const.VIDEO_BACKGROUND)
        substractBackground(camArray[i][0], const.VIDEO_TEST, model, 0)
    print("THE END")