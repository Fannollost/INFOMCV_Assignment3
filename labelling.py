from cameraCalibration import showImage, getImagesFromVideo
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

    clusters = [None] * const.CLUSTER_AMOUNT
    for i in range(len(clusters)):
        clusters[i] = d[labels.ravel()==i]

    #print(clusters)
    histograms = [None] * const.CLUSTER_AMOUNT
    frame = getFrame()
    frame = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
    #showImage('frame', frame, 0)
    orderedPos = []
    orderedCol = []
    for i in range(const.CLUSTER_AMOUNT):
        histogram = np.zeros(26) #bin count
        total = 0
        for voxel in clusters[i]:
            imgPoint = table[int(voxel[0]), int(voxel[2]), int(voxel[1])]

            heightIm = 644
            widthIm = 486
            if 0 <= imgPoint[0] < heightIm and 0 <= imgPoint[1] < widthIm:
                color = frame[imgPoint[0], imgPoint[1]]
                print(color[0])
                histogram[int(np.floor(color[2] / 10))] += 1 #check H value put it in bin
                total += 1
            #else:
            #    colors.append([0,0,0])
        histograms[i] = histogram / total
        #colors[i] = [np.argmax(histogram) * 10, 0, 0]
        if not isTrained:
            histogramModels[i] = histogram / total
        #print(histograms)
        for voxel in clusters[i]:
            orderedPos.append(voxel)
            colorModel = [(np.argmax(histograms[i]) * 10), (np.argmax(histograms[i]) * i * 10), i / 2]
            
            if isTrained:
                lowestDistance = 300000 #placeholder for distance
                index = 0
                for j in range(len(histogramModels)):
                    h = histogramModels[j]
                    distance = 0
                    for b in range(len(h)):
                        distance += abs(histogram[b] - h[b])
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
    isTrained = True
    return orderedPos, orderedCol
