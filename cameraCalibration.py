import numpy as np
import constants as const
import cv2 as cv
import glob
import math
import os.path
import xml.etree.ElementTree as ET

global clickPoints
global counter
clickPoints = []
counter = 0

#Draws the axis on the board
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    pt1 = (int(corner[0]), int(corner[1]))
    dest1 = tuple(imgpts[0].ravel())
    dest2 = tuple(imgpts[1].ravel())
    dest3 = tuple(imgpts[2].ravel())
    img = cv.line(img, pt1, (int(dest1[0]),int(dest1[1])), (255,0,0), 2)
    img = cv.line(img, pt1, (int(dest2[0]),int(dest2[1])), (0,255,0), 2)
    img = cv.line(img, pt1, (int(dest3[0]),int(dest3[1])), (0,0,255), 2)
    return img

#Draws the cube on the board
def drawCube(img, corners, imgpts):
    imgpts = (np.int32(imgpts)).reshape(-1,2)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-2)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),2)
    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),2)
    return img


#Function to show image
def showImage(name, image, wait = -1):
    cv.imshow(name, image)
    if(wait >= 0):
        cv.waitKey(wait)

#Get mouse click event
#If leftmouse click, save and print the point
def click_event(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        global counter
        clickPoints.append((x,y))
        counter += 1

#returns true if quality is good for calibration and image based on the sharpness of the chessboard
def checkQuality(gray, corners, limit):
    retval, sharp = cv.estimateChessboardSharpness(gray, const.BOARD_SIZE, corners)
    if retval[0] > limit:
        print("Sharpness : " + str(retval[0]) +" - Limit :" + str(limit) )
    return retval[0] <= limit

#Improves the quality of the chessboard by enhancing edges
def improveQuality(gray):

    #determine original sharpness of the board
    ret, corners = cv.findChessboardCorners(gray, const.BOARD_SIZE, None)
    if ret == True:
        retval, sharp = cv.estimateChessboardSharpness(gray, const.BOARD_SIZE, corners)
        print("Sharpness : " + str(retval[0]))

    #enhance the edges
    edges = cv.Canny(gray, 150, 400)
    h , w = gray.shape[:2]
    for l in range(h):
        for c in range(w):
            if(edges[l,c] > 250):
                gray[l,c] = 0
    showImage(const.WINDOW_NAME,gray,1500)

    #determine updated sharpness of the board
    ret, corners = cv.findChessboardCorners(gray, const.BOARD_SIZE, None)
    if ret == True:
        retval, sharp = cv.estimateChessboardSharpness(gray, const.BOARD_SIZE, corners)
        print("Corrected sharpness : " + str(retval[0] ))
    return gray, ret, corners

#function to draw the axis and the cube on given input frame
def drawOrigin(frame, criteria, objp, mtx, dist , webcam = False, camera = None):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if webcam :
        ret, corners = cv.findChessboardCorners(gray, const.BOARD_SIZE, cv.CALIB_CB_FAST_CHECK)
    else:
        #ret, corners = cv.findChessboardCorners(gray, const.BOARD_SIZE, None)
        imgpoints, objpoints, corners = pickCorners([],[],objp,frame,gray,criteria, False)
        ret = True

    if (ret == True):
        corners2 = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        imgpts, jac = cv.projectPoints(const.AXIS, rvecs, tvecs, mtx, dist)
        cubeimgpts, jac = cv.projectPoints(const.CUBE_AXIS, rvecs, tvecs, mtx, dist)
        img = draw(frame, corners2, imgpts)
        #img = drawCube(img, corners2, cubeimgpts)
        saveCalibration(mtx,dist,rvecs,tvecs, camera)
        return img
    else:
        return frame

def pickCorners(imgpoints, objpoints, objp, img, gray, criteria, showLines = True):
    showImage(const.WINDOW_NAME, img)
    global counter
    global clickPoints
    counter = 0
    clickPoints = []
    while(counter < 4):
        #Get mouseinput
        #the mouse clicking should be done by starting at the bottom right black corner in a horizontally rotated chessboard. If the chessboard
        #is rotated differently, the same corresponding corner should be picked 
        cv.setMouseCallback(const.WINDOW_NAME, click_event)
        cv.waitKey(1)

        #visual feedback for mouseclicks
        if(counter != 0):
            img = cv.circle(img, clickPoints[counter - 1], 5, const.RED)
            showImage(const.WINDOW_NAME, img)

    #prepare the pointset
    interpolatedPoints = np.zeros((const.BOARD_SIZE[1], const.BOARD_SIZE[0], 2))
    largest = 0
    smallest = 5000 

    #indexes
    diagonalPoint = 0
    closestPoint = 0

    #find closest and diagonal point
    for j in range(len(clickPoints)):
        dist = math.dist(clickPoints[0], clickPoints[j])
        if(dist != 0 and smallest > dist):
            smallest = dist
            closestPoint = j
        if(dist != 0 and largest < dist):
            largest = dist
            diagonalPoint = j

    #determine approximate distance between points
    shortSteps = math.dist(clickPoints[0],clickPoints[closestPoint]) / (const.BOARD_SIZE[1])
    longSteps = math.dist(clickPoints[closestPoint], clickPoints[diagonalPoint]) / (const.BOARD_SIZE[0])

    #generate uniform set of points
    interpolatedPoints[0,0] = clickPoints[0]
    orig = clickPoints[0]
    for x in range(const.BOARD_SIZE[0]):
        for y in range(const.BOARD_SIZE[1]):
            interpolatedPoints[y,x] = (orig[0] + longSteps * x, orig[1] + shortSteps * y)

    #get uniform corners      
    stepFactorX = const.BOARD_SIZE[0] - 1
    stepFactorY = const.BOARD_SIZE[1] - 1

    uniform = np.array((orig, 
    (orig[0] + longSteps * stepFactorX, orig[1] + shortSteps * 0),
    (orig[0] + longSteps * stepFactorX, orig[1] + shortSteps * stepFactorY),
    (orig[0] + longSteps * 0, orig[1] + shortSteps * stepFactorY))).astype(np.float32)
    dst = np.array(clickPoints).astype(np.float32)
  
    #transform uniform set of points to desired cornerpoints
    transform_mat = cv.findHomography(uniform,dst)[0]
    corners2 = cv.perspectiveTransform(interpolatedPoints, transform_mat)
    corners2 = np.array(corners2).reshape(const.BOARD_SIZE[0]*const.BOARD_SIZE[1],1,2).astype(np.float32)

    edges = cv.Canny(img, 150, 250)
    corners2 = cv.cornerSubPix(edges,corners2,(5, 5), (-1,-1), criteria)

    if not checkQuality(gray, corners2, 4) and const.REJECT_LOW_QUALITY:
        print("Image Rejected")
        return imgpoints, objpoints, corners2
    
    imgpoints.append(corners2)
    objpoints.append(objp)
    # Draw and display the corners
    if(showLines):
        cv.drawChessboardCorners(img, const.BOARD_SIZE, corners2, True)
        showImage(const.WINDOW_NAME, img, 100)
    return imgpoints, objpoints, corners2

def getImagesFromVideo(camera, videoType, amountOfFrames, at = -1):
    video = cv.VideoCapture(camera + videoType)
    frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    frames = []
    if(at != -1):
        video.set(cv.CAP_PROP_POS_FRAMES, at)
        ret, frame = video.read()
        return frame
    for i in range(amountOfFrames):
        frame_number = i * int(frame_count / amountOfFrames)
        video.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video.read()
        frames.append(frame)
    return frames 

def saveCalibration(mtx, dist, rvecs, tvecs, camera):
    root = ET.Element('opencv_storage')
    camMat = ET.SubElement(root, 'CameraMatrix')
    camMat.set('type_id', 'opencv-matrix')
    rows = ET.SubElement(camMat, 'rows')
    rows.text = "3"
    cols = ET.SubElement(camMat, 'cols')
    cols.text = "3"
    dt = ET.SubElement(camMat, 'dt')
    dt.text = "f"
    data = ET.SubElement(camMat, 'data')
    mtxText = ""
    for l in range(3):
        for c in range (3):
            mtxText = mtxText + str(mtx[l,c]) + " "
        mtxText = mtxText[:-1]
        mtxText = mtxText + "\n"
    data.text = mtxText


    dCoeff = ET.SubElement(root, 'DistortionCoeffs')
    dCoeff.set('type_id', 'opencv-matrix')
    rows = ET.SubElement(dCoeff, 'rows')
    rows.text = "5"
    cols = ET.SubElement(dCoeff, 'cols')
    cols.text = "1"
    dt = ET.SubElement(dCoeff, 'dt')
    dt.text = "f"
    data = ET.SubElement(dCoeff, 'data')
    dCoeffText = ""
    for l in range(5):
        dCoeffText = dCoeffText + str(dist[0,l]) +"\n"
    data.text = dCoeffText

    rvecsValue = ET.SubElement(root, 'RVecs')
    rvecsValue.set('type_id', 'opencv-matrix')
    rows = ET.SubElement(rvecsValue, 'rows')
    rows.text = "3"
    cols = ET.SubElement(rvecsValue, 'cols')
    cols.text = "1"
    dt = ET.SubElement(rvecsValue, 'dt')
    dt.text = "f"
    data = ET.SubElement(rvecsValue, 'data')
    rvecsValue = ""
    for l in range(3):
        rvecsValue = rvecsValue + str(rvecs[l,0]) +"\n"
    data.text = rvecsValue

    tvecsValue = ET.SubElement(root, 'TVecs')
    tvecsValue.set('type_id', 'opencv-matrix')
    rows = ET.SubElement(tvecsValue, 'rows')
    rows.text = "3"
    cols = ET.SubElement(tvecsValue, 'cols')
    cols.text = "1"
    dt = ET.SubElement(tvecsValue, 'dt')
    dt.text = "f"
    data = ET.SubElement(tvecsValue, 'data')
    tvecsValue = ""
    for l in range(3):
        tvecsValue = tvecsValue + str(tvecs[l,0]) +"\n"
    data.text = tvecsValue

#    xmlTxt = ET.tostring(root)
    tree = ET.ElementTree(root)

    f = open(camera+"data.xml", "wb")
    tree.write(f, encoding='utf-8', xml_declaration=True)
    f.close()
    return 0

def main(currentCam):

    camera = currentCam[0]
    const.BOARD_SIZE = currentCam[1]
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((const.BOARD_SIZE[0]*const.BOARD_SIZE[1],3), np.float32)
    objp[:,:2] = (const.SQUARE_SIZE * np.mgrid[0:const.BOARD_SIZE[0], 0:const.BOARD_SIZE[1]]).T.reshape(-1,2)
    #if no configuration file is found, or if calibration is forced, calibrate the camera
    #need to check if config is done!
    if(const.FORCE_CALIBRATION):
        #prepare object points
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real wold space
        imgpoints = [] # 2d points in image space

        images = getImagesFromVideo(camera, const.SELECTED_VIDEO, const.IMAGES_CALIB_NB)

        global counter
        global clickPoints
        for frame in images:
            img = frame
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            clickPoints = []
            counter = 0

            #find the chessboard corners
            #gray, ret, corners = improveQuality(gray)
            ret, corners = cv.findChessboardCorners(gray, const.BOARD_SIZE, None)

            #reject the low quality images
            if ret and not checkQuality(gray, corners, 5) and const.REJECT_LOW_QUALITY:
                print("Image Rejected")
                continue

            #if found, add object points, image points (after refining them)
            #if ret == True:
                corners2 = cv.cornerSubPix(gray,corners,(5,5), (-1,-1), criteria)

                imgpoints.append(corners2)
                objpoints.append(objp)

                # Draw and display the corners
                cv.drawChessboardCorners(img, const.BOARD_SIZE, corners2, ret)
                showImage(const.WINDOW_NAME, img, 300)

            else:
                pickCorners(imgpoints,objpoints,objp,img, gray, criteria)  

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error

        print("total error: {}".format(mean_error/len(objpoints)) )
        np.savez(const.DATA_PATH, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    else:
        calibration = np.load(const.DATA_PATH)
        
        #extract calibration values from the file:
        mtx = calibration['mtx']
        dist = calibration['dist']
    
    print(mtx)
    #online phase, check if the webcam functionality is on
    if(const.WEBCAM == True):
        cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        
        if not cap.isOpened():
            raise IOError("Webcam not accessible")

        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        while True:
            ret, frame = cap.read()

            #draw axis and cube on the board
            img = drawOrigin(frame, criteria, objp, mtx, dist, True)
            cv.imshow(const.WINDOW_NAME, img)

            c = cv.waitKey(1)
            if c == 27:
                break

            try :
                cv.getWindowProperty(const.WINDOW_NAME, 0)
            except :
                break
        cap.release()
    else:    
        #draw axis and cube on test image
        frames = getImagesFromVideo(camera, const.VIDEO_EXTRINSICS, 1)
        img = drawOrigin(frames[0], criteria, objp, mtx, dist, camera=camera)
        showImage(const.WINDOW_NAME, img, 0)

    cv.destroyAllWindows()



if __name__ == "__main__":
    calibration = np.load(const.DATA_PATH)
    mtx = calibration['mtx']
    dist = calibration['dist']
    camArray = [const.CAM1,const.CAM2,const.CAM3,const.CAM4]
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((const.BOARD_SIZE[0]*const.BOARD_SIZE[1],3), np.float32)
    objp[:,:2] = (const.SQUARE_SIZE * np.mgrid[0:const.BOARD_SIZE[0], 0:const.BOARD_SIZE[1]]).T.reshape(-1,2)

    for cam in camArray:
        frames = getImagesFromVideo(cam[0], const.VIDEO_EXTRINSICS, 1)
        img = drawOrigin(frames[0], criteria, objp, mtx, dist, False, cam[0])
        showImage(const.WINDOW_NAME, img, 0)