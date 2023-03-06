import numpy as np
#declare constants

#colors
RED    = (255,0,0)
ORANGE = (255, 102, 0)
YELLOW = (255, 255, 0)
GREEN  = (0,255,0)
LBLUE  = (102, 153, 255)
BLUE   = (0,0,255)

#board properties
BOARD_SIZE = (8,6)
SQUARE_SIZE = 115

#window properties
WINDOW_NAME = 'img'
WINDOW_SIZE = (60,40)

#images paths for assignment 1
DATA_PATH   = './data/calibration.npz'
IMAGES_PATH = './pics/*.jpg'
IMAGES_PATH_YANNICK = './pics/yannick/test/*.jpg'
IMAGES_PATH_FABIEN = './pics/fabien/*.jpg'
IMAGES_PATH_DEFAULT = './pics/default/*.jpg'
IMAGES_PATH_FLOOR = './pics/floor/*.jpg'
IMAGES_PATH_TEST_MANUAL = './pics/testingSet/manual*.jpg'
IMAGES_PATH_TEST_ALL = './pics/testingSet/*.jpg'
IMAGES_PATH_TEST_SELECTION = './pics/testingSet/selection*.jpg'
IMAGES_PATH_TEST_SUB_SELECTION = './pics/testingSet/selection_sub*.jpg'

#worldcoordinates indicators
AXIS = SQUARE_SIZE * np.float32( [[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
CUBE_AXIS = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0],
                   [0,0,-1],[0,1,-1],[1,1,-1],[1,0,-1] ])

#settings
WEBCAM = False
FORCE_CALIBRATION = True
REJECT_LOW_QUALITY = True

#video paths
CAM1 = ("./data/cam1/",(8,6), 0)
CAM2 = ("./data/cam2/",(8,6), 1)
CAM3 = ("./data/cam3/",(8,6), 2)
CAM4 = ("./data/cam4/",(8,6), 3)
SELECTED_CAM = CAM2

#videos
VIDEO_INTRINSICS = 'intrinsics.avi'
VIDEO_CHECKERBOARD = 'checkerboard.avi'
VIDEO_BACKGROUND = 'background.avi'
VIDEO_TEST = 'video.avi'
SELECTED_VIDEO = VIDEO_INTRINSICS
INTRINSICS_DATA = 'intrinsics.xml'
IMAGES_CALIB_NB = 20
IMAGES_BACKGROUND_NB = 25
H = 0
S = 1
V = 2
H_WEIGHT = 8
S_WEIGHT = 4
V_WEIGHT = 1 # This is shadow detection
THRESHOLD = 160
SCENE_SCALE_DIV = 40
BLOCK_SIZE = 1.0
FRAME_SIZE = 50