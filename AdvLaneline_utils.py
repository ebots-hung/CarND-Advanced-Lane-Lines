# -*- coding: utf-8 -*-
# @Author:              QH Lam
# @Date:                2021-02-27
# @Module name:         utils.py
# @Module description:  Utility functions for CarND-Advanced Lane lines

import os
import cv2
import glob
import sys
import numpy as np
import matplotlib.image as mpimg

# @function name:           Camera_Calibration
# @function description:    Do camera calibration with object points and image points
# @param:                   camera_cal_folder, output_folder, debugflag = False
# @ret:                     ret, mtx, dist, rvecs, tvecs
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
    
    # Do camera calibration with given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size,None,None)

    return ret, mtx, dist, rvecs, tvecs


# @function name:           Image_Undistortion
# @function description:    Do image undistortion, combine the test image with undistorqued image, then save to outputfolder
# @param:                   mtx, dist, imagepath, imagename, output_folder
# @ret:                     undistorted_img, newmtx
def Image_Undistortion(mtx, dist, img, output_folder, index, imgcombine = True): 
    # get cwd
    # currentfolderpath = os.path.abspath(os.getcwd())
    # imgfullpath = os.path.join(currentfolderpath, imagepath, imagename)
    # outputsavefolder = os.path.join(currentfolderpath,output_folder)
    # img = cv2.imread(imgfullpath)
    newmtx = mtx
    undistorted_img = cv2.undistort(img, mtx, dist, None, newmtx)
    if imgcombine == True:
        combineimg = cv2.hconcat([img, undistorted_img])
    else: 
        combineimg = undistorted_img
    cv2.imwrite(os.path.join(output_folder, f'undistorted_image{index}.jpg'), combineimg)
    return undistorted_img, newmtx
    
# @function name:           PerspectiveTransform_unwarp
# @function description:    Do perspective transform and unwarp the input image
# @param:                   image, src, dst (define source and destination points for transform)
# @ret:                     warped, M, M_inv
def PerspectiveTransform_unwarp(iimg, src, dst, hw):
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and inv(M), the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(iimg, M, (hw[1],hw[0]), flags=cv2.INTER_LINEAR)
    return warped, M, M_inv

# @function name:           abs_sobel_thresh
# @function description:    Define a function that applies Sobel x or y, then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img, orient='x', thresh_min=25, thresh_max=255):
    # Convert to grayscale === or LAB L channel
    gray = (cv2.cvtColor(img, cv2.COLOR_RGB2Lab))[:,:,0]
    # Take the derivative in x or y given orient = 'x' or 'y'
    sobel = cv2.Sobel(gray, cv2.CV_64F, orient=='x', orient=='y')
    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a mask of 1's where the scaled gradient magnitude 
    # is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sxbinary


# @function name:           mag_thresh
# @function description:    Define a function that applies Sobel x or y, then computes the magnitude of the gradient and applies a threshold.
def mag_thresh(img, sobel_kernel=25, mag_thresh=(25, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # Calculate the magnitude 
    mag_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
    # Create a binary mask where mag thresholds are met
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return sxbinary

# @function name:           dir_thresh
# @function description:    Define a function that applies Sobel x or y, then computes the direction of the gradient and applies a threshold.
def dir_thresh(img, sobel_kernel=7, thresh=(0, 0.09)):    
    # Apply the following steps to img
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    # Create a binary mask where direction thresholds are met
    binary_output =  np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    # Return this mask as your binary_output image
    return binary_output

# @function name:           hls_sthresh
# @function description:    Define a function that thresholds the S-channel of HLS, Use exclusive lower bound (>) and inclusive upper (<=)
def hls_sthresh(img, thresh=(125, 255)):
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Apply a threshold to the S channel
    binary_output = np.zeros_like(hls[:,:,2])
    binary_output[(hls[:,:,2] > thresh[0]) & (hls[:,:,2] <= thresh[1])] = 1
    # Return a binary image of threshold result
    return binary_output    

# @function name:           hls_lthresh
# @function description:    Define a function that thresholds the L-channel of HLS, Use exclusive lower bound (>) and inclusive upper (<=)
def hls_lthresh(img, thresh=(220, 255)):
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls_l = hls[:,:,1]
    hls_l = hls_l*(255/np.max(hls_l))
    # Apply a threshold to the L channel
    binary_output = np.zeros_like(hls_l)
    binary_output[(hls_l > thresh[0]) & (hls_l <= thresh[1])] = 1
    # Return a binary image of threshold result
    return binary_output    

# @function name:           lab_bthresh
# @function description:    Define a function that thresholds the B-channel of LAB, use exclusive lower bound (>) and inclusive upper (<=), OR the results of the thresholds (B channel should capture
# yellows)
def lab_bthresh(img, thresh=(190,255)):
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab_b = lab[:,:,2]
    # don't normalize if there are no yellows in the image
    if np.max(lab_b) > 175:
        lab_b = lab_b*(255/np.max(lab_b))
    # Apply a threshold to the L channel
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    # Return a binary image of threshold result
    return binary_output    

# @function name:           advlaneline_sliding_window_polyfit
# @function description:    Define method to fit polynomial to binary image with lines extracted, using sliding window
def advlaneline_sliding_window_polyfit(img):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    quarter_point = np.int(midpoint//2)
    # Previously the left/right base was the max of the left/right half of the histogram
    # this changes it so that only a quarter of the histogram (directly to the left/right) is considered
    leftx_base = np.argmax(histogram[quarter_point:midpoint]) + quarter_point
    rightx_base = np.argmax(histogram[midpoint:(midpoint+quarter_point)]) + midpoint
    
    #print('base pts:', leftx_base, rightx_base)

    # Choose the number of sliding windows
    nwindows = 15
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 40
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Rectangle data for visualization
    rectangle_data = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    left_fit, right_fit = (None, None)
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)
    
    visualization_data = (rectangle_data, histogram)
    
    return left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data    

# @function name:           advlaneline_polyfit_using_prev_fit
# @function description:    Define method to fit polynomial to binary image based upon a previous fit (chronologically speaking);
#                           this assumes that the fit will not change significantly from one video frame to the next
def advlaneline_polyfit_using_prev_fit(binary_warped, left_fit_prev, right_fit_prev):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 80
    left_lane_inds = ((nonzerox > (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] - margin)) & 
                      (nonzerox < (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] - margin)) & 
                       (nonzerox < (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    left_fit_new, right_fit_new = (None, None)
    if len(leftx) != 0:
        # Fit a second order polynomial to each
        left_fit_new = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit_new = np.polyfit(righty, rightx, 2)
    return left_fit_new, right_fit_new, left_lane_inds, right_lane_inds

# @function name:           measure_curvature_distance
# @function description:    Method to calculate radius of curvature and distance from lane center 
#                           based on binary image, polynomial fit, and L and R lane pixel indices
def measure_curvature_distance(bin_img, l_fit, r_fit, l_lane_inds, r_lane_inds):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720     # meters per pixel in y dimension
    xm_per_pix = 3.7/378    # meters per pixel in x dimension, standard lane width 12 ft = 3.7m
    left_curverad, right_curverad, center_dist = (0, 0, 0)
    # Define y-value where we want radius of curvature
    h = bin_img.shape[0]
    y = np.linspace(0, h-1, h)
    y_eval = np.max(y)
  
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = bin_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Extract left and right line pixel positions
    leftx = nonzerox[l_lane_inds]
    lefty = nonzeroy[l_lane_inds] 
    rightx = nonzerox[r_lane_inds]
    righty = nonzeroy[r_lane_inds]
    
    if len(leftx) != 0 and len(rightx) != 0:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radius of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    if r_fit is not None and l_fit is not None:
        l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
        r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
        lane_center_pos = (r_fit_x_int + l_fit_x_int) /2
        car_pos = bin_img.shape[1]/2
        center_dist = (car_pos - lane_center_pos) * xm_per_pix
    return left_curverad, right_curverad, center_dist


# @function name:           laneline_plot
# @function description:    Plot the Detected Lane into the Original Image 
def laneline_plot(original_img, binary_img, l_fit, r_fit, Minv, curv_rad, center_dist):
    if l_fit is None or r_fit is None:
        return original_img
    # Create an image to plot laneline
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    h,w = binary_img.shape
    y = np.linspace(0, h-1, num=h)                      # to cover same y-range as image
    left_fitx = l_fit[0]*y**2 + l_fit[1]*y + l_fit[2]
    right_fitx = r_fit[0]*y**2 + r_fit[1]*y + r_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=10)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=10)

    # Warp the blank back to original image space using inverse perspective matrix
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h)) 
    # Combined lane line plot back onto original image
    combined_img = cv2.addWeighted(original_img, 1, newwarp, 0.5, 0)
    
    # Add text into notification area
    widget_w = 600
    widget_h = 150
    widget = np.copy(combined_img[:widget_h, :widget_w])
    widget //= 2
    widget[0,:] = [0, 0, 255]
    widget[-1,:] = [0, 0, 255]
    widget[:,0] = [0, 0, 255]
    widget[:,-1] = [0, 0, 255]
    combined_img[:widget_h, :widget_w] = widget
    
    # Add left/right/straight curvature
    left_curve_img = mpimg.imread('./disp_images/left.png')
    right_curve_img = mpimg.imread('./disp_images/right.png')
    keep_straight_img = mpimg.imread('./disp_images/straight.png')
    left_curve_img = cv2.normalize(src=left_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    right_curve_img = cv2.normalize(src=right_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    keep_straight_img = cv2.normalize(src=keep_straight_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # curvature direction
    value = None

    if abs(l_fit[0]) > abs(r_fit[0]):
        value = l_fit[0]
    else:
        value = r_fit[0]
    # print(value, l_fit[0], r_fit[0])
    msg = "Keep Straight Ahead"
    if abs(value) <= 0.000015:
        px, py = keep_straight_img[:,:,3].nonzero()    #get straight image - nonzero
        combined_img[px+25, py+450] = keep_straight_img[px, py, :3]
        msg = "Keep Straight Ahead"
    elif value < 0:
        px, py = left_curve_img[:,:,3].nonzero()        #get left curve image - nonzero
        combined_img[px+25, py+450] = left_curve_img[px, py, :3]
        msg = "Left Curve Ahead"
    else:
        px, py = right_curve_img[:,:,3].nonzero()       #get right image - nonzero
        combined_img[px+25, py+450] = right_curve_img[px, py, :3]
        msg = "Right Curve Ahead"

    radtext = 'Curve radius: ' + '{:04.2f}'.format(curv_rad) + 'm'
    cv2.putText(combined_img, radtext, org=(40,60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
    direction = ''
    if center_dist > 0:
        direction = 'right'
    elif center_dist < 0:
        direction = 'left'
    centertext = '{:04.3f}'.format(abs(center_dist)) + 'm ' + direction + ' of center'
    cv2.putText(combined_img, centertext, org=(40,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)    
    return combined_img

# @function name:           image_pipeline
# @function description:    Define the complete image processing pipeline, read raw image and return binary image with lane lines identified
def image_pipeline(method, mtx, dist, index, dump_image_folderpath, img, debug_folder, debug_flag = False): 
    # Undistort
    img_undistort,__ = Image_Undistortion(mtx, dist, img, dump_image_folderpath, index, False)  

    srcpts = np.float32([(580,466),
                  (707,466), 
                  (259,683), 
                  (1050,683)])
    dstpts = np.float32([(450,0),
                  (830,0),
                  (450,720),
                  (830,720)])
    # Perspective Transform
    img_unwarp, M, Minv = PerspectiveTransform_unwarp(img_undistort, srcpts, dstpts, img_undistort.shape)
    if debug_flag == True: 
        cv2.imwrite(os.path.join(debug_folder,f'unwarped_pipeline{index}.jpg'), img_unwarp)

    if method == ("Sobel_Absolute"): 
        # Sobel Absolute (using default parameters)
        # print("Sobel_Absolute")
        img_sobelAbs = abs_sobel_thresh(img_unwarp)
        combined = img_sobelAbs
    elif method == ("Sobel_Magnitude"): 
        # Sobel Magnitude (using default parameters)
        img_sobelMag = mag_thresh(img_unwarp)
        combined = img_sobelMag
    elif method == ("Sobel_Direction"):         
        # Sobel Direction (using default parameters)
        img_sobelDir = dir_thresh(img_unwarp)
        combined = img_sobelDir
    elif method == ("Sobel_MagDir"): 
        img_sobelMag = mag_thresh(img_unwarp)
        img_sobelDir = dir_thresh(img_unwarp)
        combined = np.zeros_like(img_sobelMag)
        combined[(img_sobelMag == 1) | (img_sobelDir == 1)] = 1
    elif method == ("HLS_S_channel"):         
        # HLS S-channel Threshold (using default parameters)
        img_SThresh = hls_sthresh(img_unwarp)
        combined = img_SThresh
    elif method == ("HLS_L_channel"):
        # HLS L-channel Threshold (using default parameters)
        img_LThresh = hls_lthresh(img_unwarp)
        combined = img_LThresh
        # cv2.imwrite(os.path.join(debug_folder,f'img_LThresh{index}.jpg'), np.multiply(img_LThresh,255))
    elif method == ("LAB_B_channel"):
        # Lab B-channel Threshold (using default parameters)
        img_BThresh = lab_bthresh(img_unwarp)
        combined = img_BThresh
    elif method == ("HLSCombine"):    
        # Combine HLS and Lab B channel thresholds
        # print("HLSCombine")
        img_LThresh = hls_lthresh(img_unwarp)
        img_BThresh = lab_bthresh(img_unwarp)
        combined = np.zeros_like(img_BThresh)
        combined[(img_LThresh == 1) | (img_BThresh == 1)] = 1
    # Debug: write grayscale images into jpg files
    grayscale_img = np.multiply(combined, 255)
    if debug_flag == True: 
        cv2.imwrite(os.path.join(debug_folder,f'grayscale_img{index}.jpg'), grayscale_img)
    return combined, Minv