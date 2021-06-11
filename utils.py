# -*- coding: utf-8 -*-
# @Author:              QH Lam
# @Date:                2021-02-27
# @Module name:         utils.py
# @Module description:  Utility functions for CarND-Advanced Lane lines

import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

def Camera_Calibration(camera_cal_folder, output_folder, debugflag = False):
    # get cwd
    currentfolderpath = os.path.abspath(os.getcwd())
    camcalpath = os.path.join(currentfolderpath,camera_cal_folder)
    outputfolder = os.path.join(currentfolderpath,output_folder)
    # print debug info
    if debugflag == True:
        print("[Debug]Current folder path:", currentfolderpath)
        print("Images folder path:", camcalpath)
        print("Output folder path:", outputfolder)

    # Ref: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html     
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    image_size = 0

    lst_cal_images = glob.glob(os.path.join(camcalpath,'calibration*.jpg'))

    # print debug info
    if debugflag == True:
        for imagefile in lst_cal_images: 
            print(imagefile)

    # Step through the list and search for chessboard corners
    for i, fname in enumerate(lst_cal_images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if i == 0: image_size = (img.shape[1], img.shape[0])
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
        if ret == True: 
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            dump_img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            cv2.imwrite(os.path.join(outputfolder, f'output_image{i}.jpg'), dump_img)
        # else: 
        #     dump_img = img
        #     cv2.imwrite(os.path.join(outputfolder, f'notfoundcorner_image{i}.jpg'), dump_img)
    
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size,None,None)

    return ret, mtx, dist, rvecs, tvecs

def Image_Undistortion(mtx, dist, imagepath, imagename, output_folder): 
    # get cwd
    currentfolderpath = os.path.abspath(os.getcwd())
    imgfullpath = os.path.join(currentfolderpath, imagepath, imagename)
    outputsavefolder = os.path.join(currentfolderpath,output_folder)
    img = cv2.imread(imgfullpath)
    newmtx = mtx
    undistorted_img = cv2.undistort(img, mtx, dist, None, newmtx)
    combineimg = cv2.hconcat([img, undistorted_img])
    cv2.imwrite(os.path.join(outputsavefolder, 'undistorted_image.jpg'), combineimg)
    return undistorted_img, newmtx
    