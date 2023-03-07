from cameraCalibration import showImage, getImagesFromVideo
import constants as const
import cv2 as cv
import numpy as np

def color():
    #get a good frame from camera3
    frame = getImagesFromVideo(const.CAM3[0],'video.avi', 1, 480)
    showImage('frame', frame, 0)

def getCenters():
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_PP_CENTERS
    data = np.load(const.CLUSTER_PATH)
    data = data['data']
    data = data[:,:2]
    data = np.float32(data)
    compactness, labels, center = cv.kmeans(data,3,None, criteria, 10, flags)

    #clusters, for now 3, since we dont have the 4th person in yet. 
    print("first" + str(center[0]) + str(data[labels.ravel()==0]))
    print("second"+ str(center[1]) + str(data[labels.ravel()==1]))
    print("third" + str(center[2]) + str(data[labels.ravel()==2]))
    
getCenters()