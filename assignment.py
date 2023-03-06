import glm
import random
import numpy as np
import xml.etree.ElementTree as ET
import constants as const
import cv2 as cv
from engine.config import config
from background import get_background_model, get_foreground_mask, get_difference
from marchingCube import marchingCube

#initialize 'normal' lookup table
global tables
global tableInitalized
tables = [0,0,0,0]
tableInitialized = False

#initialize the 'xor' lookup table
global imgTables
global imgTablesInitialized
imgTables = [0,0,0,0]
imgTablesInitialized = False

global tableSaved
tableSaved = False

global prevPositions
prevPositions = []

def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data = []
    for x in range(width):
        for z in range(depth):
            data.append([x*const.BLOCK_SIZE - width/2, -const.BLOCK_SIZE, z*const.BLOCK_SIZE - depth/2])
    return data

#function to retrieve our stored xml data
def getDataFromXml(filePath, nodeName):
    tree = ET.parse(filePath)
    root = tree.getroot()
    data = ''
    for node in root.findall(nodeName):
        data = node.find("data").text
    data = data.split('\n')
    retData = []
    for i in range(len(data) - 1):
        lineData = data[i].split(' ')
        if len(lineData) > 1:
            lineRes = []
            for j in range(len(lineData)):
                lineRes.append(float(lineData[j]))
            retData.append(lineRes)
        else:
            retData.append(float(data[i]))
    return np.array(retData)

#Generate a Voxel world from camera informations
# TODO: You need to calculate proper voxel arrays instead of random ones.
def set_voxel_positions(width, height, depth, frame):
    camArray = [const.CAM1, const.CAM2, const.CAM3, const.CAM4]
    global tables
    global tableInitialized
    global prevPositions
    global tableSaved
    camParams = []
    
    for cam in camArray:
        #initialize dummy data for the lookuptables
        if tableInitialized == False:
            tables[cam[2]] = np.full(shape = (config['world_width'],config['world_height'],config['world_depth'], 2), fill_value= [-1,-1])
            
            #train the background model
            get_background_model(cam)

        #initialize camera parameters
        camPath = cam[0]
        foreground = get_foreground_mask(cam, frame)
        rvec = getDataFromXml(camPath + 'data.xml', 'RVecs')
        tvec = getDataFromXml(camPath + 'data.xml', 'TVecs')
        cameraMatrix = getDataFromXml(camPath + 'data.xml', 'CameraMatrix')
        distCoeffs = getDataFromXml(camPath + 'data.xml', 'DistortionCoeffs')
        params  = dict(rvec = rvec, tvec = tvec, cameraMatrix = cameraMatrix, distCoeffs = distCoeffs, foreground = foreground)
        camParams.append(params)
    
    data = []
    if(const.FORCE_CALIBRATION == False and tableInitialized == False):
        save = np.load(const.TABLE_PATH)
        tables = save['table']
    #instantiate grid of voxels. loop over all voxels
    voxels = np.full(shape=(config['world_width'], config['world_height'], config['world_depth']), fill_value=False)
    for x in range(width):
        print(str(100*(x+1)/width) + " %")
        for y in range(depth):
            for z in range(height):
                isOn = True
                for i in range(len(camArray)):
                    params = camParams[i]
                    table = tables[i]
                    imgpoints = []
                    
                    #if the lookup table is not initialized, we project voor each voxel the imgpoint and store it in the lookup table
                    if(tableInitialized == False and const.FORCE_CALIBRATION):
                        coord = ( (x - width / 2) * const.SCENE_SCALE_DIV, (y - depth / 2) * const.SCENE_SCALE_DIV, -z* const.SCENE_SCALE_DIV )
                        imagepoints, _ = cv.projectPoints(coord, params["rvec"], params["tvec"], params["cameraMatrix"],
                                                    params["distCoeffs"])
                        imagepoints = np.reshape(imagepoints, 2)
                        imagepoints = imagepoints[::-1]
                        table[x,y,z] = imagepoints
                    else:
                        imagepoints = table[x,y,z]
                            
    
                    #get the foreground mask of the camera. if the pixel is on for every camera, store the pixel.
                    foreground = params["foreground"]
                    (heightIm, widthIm) = foreground.shape
                    if 0 <= imagepoints[0] < heightIm and 0 <= imagepoints[1] < widthIm:
                        pixVal = foreground[int(imagepoints[0]), int(imagepoints[1])]
                        if pixVal == 0:
                           isOn = False
                    else:
                        isOn = False
                if isOn:
                    data.append([x * const.BLOCK_SIZE - width / 2, z * const.BLOCK_SIZE , y * const.BLOCK_SIZE - depth / 2])
                    voxels[x,y,z] = True

    if(tableSaved == False and const.FORCE_CALIBRATION == True):
        np.savez(const.TABLE_PATH, table=tables)
        tableSaved = True

    tableInitialized = True

    prevPositions = data
    print("Start Marching Cube")
    marchingCube(voxels)
    print("End Marching Cube")
    return data

def set_voxel_positions_xor(width,height,depth,frame):
    #calculate difference in background extraction
    global imgTables, imgTablesInitialized, tables, prevPositions
    camArray = [const.CAM1, const.CAM2, const.CAM3, const.CAM4]
    camParams =[]
    for cam in camArray:
        camPath = cam[0]
        if imgTablesInitialized == False:
            #initialize dummy values for the image lookup table
            imgTables[cam[2]] = np.full(shape = (644, 486, 3), fill_value= [-1,-1,-1])
            
        rvec = getDataFromXml(camPath + 'data.xml', 'RVecs')
        tvec = getDataFromXml(camPath + 'data.xml', 'TVecs')
        cameraMatrix = getDataFromXml(camPath + 'data.xml', 'CameraMatrix')
        distCoeffs = getDataFromXml(camPath + 'data.xml', 'DistortionCoeffs')
        params  = dict(rvec = rvec, tvec = tvec, cameraMatrix = cameraMatrix, distCoeffs = distCoeffs)
        camParams.append(params)

    if imgTablesInitialized == False:
        print(str(width) + ' ' + str(height) + ' ' + str(depth))
        #only for the second frame, we need to calculate and store the voxels corresponding to each image coordinate.
        #we do this by looping over the original voxel table, and appending each voxel to its corresponding projection point
        for cam in camArray:
            for x in range(width):
                for y in range(depth):
                    for z in range(height):
                        imgTable = imgTables[cam[2]]
                        table = tables[cam[2]]
                        correspondingPixel = table[x,y,z]
                        heightIm = 644
                        widthIm = 486
                        if 0 <= correspondingPixel[0] < heightIm and 0 <= correspondingPixel[1] < widthIm:
                            np.append(imgTable[correspondingPixel[0],correspondingPixel[1]], (x,y,z))
            imgTables[cam[2]] = imgTable
        imgTablesInitialized = True

    newVoxelsOn = []
    newVoxelsOff = []
    for cam in camArray:
        pixelsOff, pixelsOn, res = get_difference(cam,frame)
        table = imgTables[cam[2]]
        #turn the found projection points into voxel coordinates. Either remove them from the 'on' voxels or append them
        voxelsOn = [table[coord[1],coord[0]] for coord in pixelsOn]
        voxelsOff = [table[coord[1],coord[0]] for coord in pixelsOff]
        onVoxels = [[voxelCoord[0] * const.BLOCK_SIZE - width / 2, voxelCoord[2] * const.BLOCK_SIZE , voxelCoord[1] * const.BLOCK_SIZE - depth / 2] for voxelCoord in voxelsOn]
        offVoxels = [[voxelCoord[0] * const.BLOCK_SIZE - width / 2, voxelCoord[2] * const.BLOCK_SIZE , voxelCoord[1] * const.BLOCK_SIZE - depth / 2] for voxelCoord in voxelsOff]
        if newVoxelsOn == []:
            newVoxelsOn.append(onVoxels)
        else:
            newVoxelsOn = [i for i in onVoxels if i in newVoxelsOn]

        newVoxelsOff.append(offVoxels)

    for vox in newVoxelsOff:
        if vox in prevPositions:
            prevPositions.remove(vox)

    pos = prevPositions + newVoxelsOn
    return pos

# Generates dummy camera locations at the 4 corners of the room
# TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
def get_cam_positions():
    camArray = [const.CAM1, const.CAM2, const.CAM3, const.CAM4]
    cam_positions = []
    for i in range(len(camArray)):
        # calculate the real world camera positions.
        rvec = getDataFromXml(camArray[i][0] + 'data.xml', 'RVecs')
        rotationMatrix = cv.Rodrigues(np.array(rvec).astype(np.float32))[0]
        tvec = getDataFromXml(camArray[i][0] + 'data.xml', 'TVecs')
        camPos = -np.matrix(rotationMatrix).T * np.matrix(np.array(tvec).astype(np.float32)).T
        cam_positions.append([camPos[0]/const.SCENE_SCALE_DIV, -camPos[2]/const.SCENE_SCALE_DIV, camPos[1]/const.SCENE_SCALE_DIV])
    return cam_positions

# Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
# TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
def get_cam_rotation_matrices():
    camArray = [const.CAM1, const.CAM2, const.CAM3, const.CAM4]
    cam_rotations = []
    for i in range(len(camArray)):
        rvec = getDataFromXml(camArray[i][0] + '/data.xml', 'RVecs')
        rotationMatrix = cv.Rodrigues(np.array(rvec).astype(np.float32))[0]
        #calculate the camera matrices
        rotationMatrix = rotationMatrix.transpose()
        rotationMatrix = [rotationMatrix[0], rotationMatrix[2], rotationMatrix[1]]
        cam_rotations.append(glm.mat4(np.matrix(rotationMatrix).T))

    for c in range(len(cam_rotations)):
        #transform from radians to degrees.
        cam_rotations[c] = glm.rotate(cam_rotations[c], -np.pi/2 , [0, 1, 0])
    return cam_rotations


get_cam_positions()