# -*- coding: utf-8 -*-
# @Author:              QH Lam
# @Date:                2021-06-09
# @Module name:         test_CameraCalibration.py
# @Module description:  test Camera Calibration module

import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import Camera_Calibration, Image_Undistortion


#camera cal folder is at same level with test_CameraCalibration.py, utils.py
camera_cal_folderpath = "camera_cal"
dump_image_folderpath = "output_images"
test_image = "calibration1.jpg"
if __name__ == '__main__':
    print(" Test Camera Calibration")

    #run Camera_Calibration with no debug info
    ret, mtx, dist, rvecs, tvecs = Camera_Calibration(camera_cal_folderpath, dump_image_folderpath, False)

    undist_img, newmtx = Image_Undistortion(mtx, dist, camera_cal_folderpath, test_image, dump_image_folderpath)

    calib_data = {}
    # calib_data["Rot"] = rvecs
    # calib_data["Trans"] = tvecs
    calib_data["mtx"] = newmtx
    calib_data["dist"] = dist
    currentfolderpath = os.path.abspath(os.getcwd())
    filetowrite = os.path.join(currentfolderpath, dump_image_folderpath, "camcalibration.ini")
 
    with open( filetowrite, "w" ) as file: 
        for key, value in calib_data.items(): 
            file.write('%s = %s\n' % (key, value))
