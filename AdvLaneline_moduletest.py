# -*- coding: utf-8 -*-
# @Author:              QH Lam
# @Date:                2021-06-09
# @Module name:         test_CameraCalibration.py
# @Module description:  test Camera Calibration module

import os
import cv2
import glob
import numpy as np

from AdvLaneline_utils import Camera_Calibration, Image_Undistortion, PerspectiveTransform_unwarp, image_pipeline, advlaneline_sliding_window_polyfit, advlaneline_polyfit_using_prev_fit, measure_curvature_distance, laneline_plot


#camera cal folder is at same level with test_CameraCalibration.py, utils.py
camera_cal_folderpath = "camera_cal"
dump_image_folderpath = "output_images"
undistorted_input_image_name = "calibration1.jpg"
debug_flag = True
if __name__ == '__main__':
    print(" Test Camera Calibration")
    currentfolderpath = os.path.abspath(os.getcwd())
    # [1]: run Camera_Calibration with no debug info
    ret, mtx, dist, rvecs, tvecs = Camera_Calibration(camera_cal_folderpath, dump_image_folderpath, False)

    # [2]: run Camera undistortion 
    undistorted_input_image = cv2.imread(os.path.join(currentfolderpath, camera_cal_folderpath, undistorted_input_image_name))
    output_image_folder = os.path.join(currentfolderpath, dump_image_folderpath)
    undist_img, newmtx = Image_Undistortion(mtx, dist, undistorted_input_image, output_image_folder, 'x0', True)

    calib_data = {}
    # calib_data["Rot"] = rvecs
    # calib_data["Trans"] = tvecs
    calib_data["mtx"] = newmtx
    calib_data["dist"] = dist
    currentfolderpath = os.path.abspath(os.getcwd())
    filetowrite = os.path.join(currentfolderpath, dump_image_folderpath, "camcalibration.ini")
 
    with open( filetowrite, "w" ) as file: 
        for key, value in calib_data.items(): 
            # print(dist)
            file.write('%s=\n%s\n' % (key, value))

    # [3]: test image with perspective transform and unwarp
    test_image_folderpath = "test_images"
    undistorted_input_image = cv2.imread(os.path.join(currentfolderpath, test_image_folderpath, "test2.jpg"))
    undist_img,__ = Image_Undistortion(mtx, dist, undistorted_input_image, dump_image_folderpath, 1, False)

    # srcpts = np.float32([(550, 460),    # top-left
    #                 (150, 720),         # bottom-left
    #                 (1200, 720),        # bottom-right
    #                 (770, 460)])        # top-right
    # dstpts = np.float32([(100, 0),
    #                 (100, 720),
    #                 (1100, 720),
    #                 (1100, 0)])   
    

    srcpts = np.float32([(580,466),
                  (707,466), 
                  (259,683), 
                  (1050,683)])
    dstpts = np.float32([(450,0),
                  (830,0),
                  (450,720),
                  (830,720)])     
    hw = undist_img.shape
    # print(hw)
    exampleImg_unwarp, M, Minv = PerspectiveTransform_unwarp(undist_img, srcpts, dstpts, hw)
    cv2.imwrite(os.path.join(currentfolderpath, dump_image_folderpath, "unwarped_image.jpg"), exampleImg_unwarp)
    pts = np.array([[258,682],[575,464],[727,464],[1089,682]], np.int32)
    pts = pts.reshape((-1, 1, 2))  
    undist_img_polylines = cv2.polylines(undist_img, [pts], True, (0,0,255), 3)
    cv2.imwrite(os.path.join(currentfolderpath, dump_image_folderpath, "polylines_image.jpg"), undist_img_polylines)

    # [4]: run image pipeline process for all test images
    lst_test_images = glob.glob(os.path.join(currentfolderpath, test_image_folderpath, '*.jpg'))
    debug_folder = os.path.join(currentfolderpath,"debug")
    for idx, image in enumerate(lst_test_images): 
        iimg0 = cv2.imread(image)
        iimg1 = cv2.cvtColor(iimg0, cv2.COLOR_BGR2RGB)
        if debug_flag == True:
            iimg2 = cv2.hconcat([iimg0, iimg1])
            cv2.imwrite(os.path.join(debug_folder,f'debug{idx}.jpg'), iimg2)
        oimg, Minv = image_pipeline("HLSCombine", mtx, dist, idx, os.path.join(currentfolderpath, dump_image_folderpath), iimg1, debug_folder)
        cv2.imwrite(os.path.join(output_image_folder, f'pipeline_img{idx}.jpg'), np.multiply(oimg,255))
    

    # [5]: run test for sliding window polyfit
    img_test_file = os.path.join(currentfolderpath, dump_image_folderpath, 'pipeline_img3.jpg')
    iimg0 = cv2.imread(img_test_file, cv2.IMREAD_GRAYSCALE)
    # print(type(iimg0),iimg0)
    lfit, rfit, llane_inds, rlane_inds, visualization_data = advlaneline_sliding_window_polyfit(iimg0)
    # Create an output image to draw on and visualize the result
    oimg1 = np.uint8(np.dstack((iimg0, iimg0, iimg0))*255)
    for rect in visualization_data[0]:
        cv2.rectangle(oimg1,(rect[2],rect[0]),(rect[3],rect[1]),(0,255,0), 2) 
        cv2.rectangle(oimg1,(rect[4],rect[0]),(rect[5],rect[1]),(0,255,0), 2) 
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = iimg0.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    oimg1[nonzeroy[llane_inds], nonzerox[llane_inds]] = [255, 0, 0]
    oimg1[nonzeroy[rlane_inds], nonzerox[rlane_inds]] = [100, 200, 255]
    # Update for lane lines
    pixely = np.linspace(0, iimg0.shape[0]-1, iimg0.shape[0])
    # print(iimg0.shape[0])
    # print(pixely)
    for y in pixely: 
        lpixelx = np.int32(lfit[0]*y**2 + lfit[1]*y + lfit[2])
        oimg1[np.int32(y), lpixelx] = [255, 255, 0]
        rpixelx = np.int32(rfit[0]*y**2 + rfit[1]*y + rfit[2])
        oimg1[np.int32(y), rpixelx] = [255, 255, 0]
        # print(lpixelx, rpixelx, np.int32(y))
    cv2.imwrite(os.path.join(currentfolderpath, dump_image_folderpath, 'test_advlaneline_sliding_window_polyfit.jpg'),oimg1)
    print(lfit)
    print(rfit)
    print(len(llane_inds))
    print(rlane_inds)

    # [6]: run test for radius/curvature calc and laneline_plot function
    lR, rR, pos = measure_curvature_distance(iimg0, lfit, rfit, llane_inds, rlane_inds)
    result_img = laneline_plot(undistorted_input_image, iimg0, lfit, rfit, Minv, (lR+rR)/2, pos)
    cv2.imwrite(os.path.join(currentfolderpath, dump_image_folderpath, 'lanelinedetection.jpg'),result_img)