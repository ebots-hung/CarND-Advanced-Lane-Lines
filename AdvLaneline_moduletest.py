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
from AdvLaneline_utils import Camera_Calibration, Image_Undistortion, PerspectiveTransform_unwarp


#camera cal folder is at same level with test_CameraCalibration.py, utils.py
camera_cal_folderpath = "camera_cal"
dump_image_folderpath = "output_images"
test_image = "calibration1.jpg"
if __name__ == '__main__':
    print(" Test Camera Calibration")

    #run Camera_Calibration with no debug info
    ret, mtx, dist, rvecs, tvecs = Camera_Calibration(camera_cal_folderpath, dump_image_folderpath, False)

    undist_img, newmtx = Image_Undistortion(mtx, dist, camera_cal_folderpath, test_image, dump_image_folderpath, 0, True)

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

    #test image with perspective transform and unwarp
    test_image_folderpath = "test_images"
    test_image_name = "test1.jpg"
    undist_img,__ = Image_Undistortion(mtx, dist, test_image_folderpath, test_image_name, dump_image_folderpath, 1, False)

    h,w = undist_img.shape[:2]
    # print(h)
    # print(w)
    # define source and destination points for transform
    # srcpts = np.float32([(575,464), 
    #                 (727,464),
    #                 (258,682), 
    #                 (1089,682)])
    # dstpts = np.float32([(450,0),
    #                 (w-450,0),
    #                 (450,h),
    #                 (w-450,h)])
    srcpts = np.float32([(550, 460),    # top-left
                    (150, 720),         # bottom-left
                    (1200, 720),        # bottom-right
                    (770, 460)])        # top-right
    dstpts = np.float32([(100, 0),
                    (100, 720),
                    (1100, 720),
                    (1100, 0)])         
    # print(srcpts)
    # print(pts)
    exampleImg_unwarp, M, Minv = PerspectiveTransform_unwarp(undist_img, srcpts, dstpts)
    cv2.imwrite(os.path.join(currentfolderpath, dump_image_folderpath, "unwarped_image.jpg"), exampleImg_unwarp)
    pts = np.array([[258,682],[575,464],[727,464],[1089,682]], np.int32)
    pts = pts.reshape((-1, 1, 2))  
    undist_img_polylines = cv2.polylines(undist_img, [pts], True, (0,0,255), 3)
    cv2.imwrite(os.path.join(currentfolderpath, dump_image_folderpath, "polylines_image.jpg"), undist_img_polylines)
