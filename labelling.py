from cameraCalibration import showImage, getImagesFromVideo
from engine.config import config
import constants as const
import cv2 as cv
import numpy as np


global imgTable
imgtable =[0,0,0,0]

global colorModels, histogramModels
histogramModels = [None] * const.CLUSTER_AMOUNT
colorModels = [None] * const.CLUSTER_AMOUNT

global isTrained
isTrained = False

def getFrame():
    #get a good frame from camera2
    frame = getImagesFromVideo(const.CAM2[0],'video.avi', 1, 1)
    #showImage('frame', frame, 0)
    return frame

def getCenters(data):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_PP_CENTERS

    temp = data[:,[0,2]]
    temp = np.float32(temp)
    compactness, labels, center = cv.kmeans(temp,const.CLUSTER_AMOUNT,None, criteria, 10, flags)

    #clusters, for now 3, since we dont have the 4th person in yet. 
    #print("first" + str(center[0]) + "kk" + str(data[labels.ravel()==0]))
    #print("second"+ str(center[1]) + "kk" + str(data[labels.ravel()==1]))
    #print("third" + str(center[2]) + "kk" + str(data[labels.ravel()==2]))

    #print(labels.ravel())
    
    return data, center, labels
    
#initial
def getColors(cam, data):

    global colorModels, isTrained, histogramModels
    #data = np.load(const.CLUSTER_PATH)
    #data = data['data']
    d, centers, labels = getCenters(data)
    savedTable = np.load(const.TABLE_PATH)
    lookupTable = savedTable['table']
    table = lookupTable[cam]

    width = config['world_width']
    height = config['world_height']
    depth = config['world_depth']

    clusters = [None] * const.CLUSTER_AMOUNT
    for i in range(len(clusters)):
        clusters[i] = d[labels.ravel()==i]

    #print(clusters)
    histograms = [None] * const.CLUSTER_AMOUNT
    frame = getFrame()
    frame = cv.cvtColor(frame, cv.COLOR_RGB2HSV)

    orderedPos = []
    orderedCol = []
    for i in range(const.CLUSTER_AMOUNT):
        histogram = np.zeros((26,3)) #bin count
        total = 0
        (heightIm, widthIm, _) = frame.shape
        debug = np.zeros((heightIm, widthIm, 3))

        for voxel in clusters[i]:
            imgPoint = table[int(voxel[0]+width/2), int(voxel[2]+depth/2), int(voxel[1])]
            if int(heightIm / 4) <= imgPoint[0] < (2*heightIm/4) and 0 <= imgPoint[1] < widthIm  :
                color = frame[imgPoint[0], imgPoint[1]]
                debug[imgPoint[0], imgPoint[1]] = color # cv.cvtColor(np.array([[color]]), cv.COLOR_HSV2RGB)[0,0]
                for j in range(3):
                    histogram[int(np.floor(color[j] / 10)),j] += 1 #check H value put it in bin
                total += 1
            #else:
            #    colors.append([0,0,0])
        cv.imwrite("./data/debug"+str(i)+".png", debug)
        histograms[i] = histogram / total
        #colors[i] = [np.argmax(histogram) * 10, 0, 0]
        if not isTrained:
            histogramModels[i] = histograms[i]
        #print(histograms)
        for voxel in clusters[i]:
            orderedPos.append(voxel)
            # What is the next line computing as histograms is only based on hue
            # colorModel = [(np.argmax(list(item[0] for item in histograms[i])) * 10) / 255, (np.argmax(list(item[0] for item in histograms[i])) * i * 10) / 255, i / 2]
            maxColor = np.uint8([[[(np.argmax(list(item[0] for item in histogram)) * 10), (np.argmax(list(item[1] for item in histogram))* 10), (np.argmax(list(item[2] for item in histogram))* 10)]]])
            #colorModel = cv.cvtColor(maxColor, cv.COLOR_HSV2RGB)
            colorModel = maxColor[0,0] /255
            if isTrained:
                lowestDistance = 300000 #placeholder for distance
                index = 0
                for j in range(len(histogramModels)):
                    h = histogramModels[j]
                    distance = 0
                    for b in range(len(h)):
                        for i in range(3):
                            distance += abs(histogram[b,i] - h[b,i])
                    if distance < lowestDistance:
                        lowestDistance = distance
                        index = j
                original = colorModels[index]
                #needto track a list of centers
                colorModels[index] = (centers[i], original[1])
                orderedCol.append(original[1])
            else:
                orderedCol.append(colorModel)
                colorModels[i] = (centers[i], colorModel)
                # colorModels = [(centers[0], [255, 0, 0]), (centers[1], [0, 255, 0]), (centers[2], [0, 0, 255]),
                #               (centers[3], [255, 255, 0])]
                # orderedCol = [ [255,0,0], [0,255,0], [0,0,255], [255,255,0]]
    isTrained = True
    return orderedPos, orderedCol