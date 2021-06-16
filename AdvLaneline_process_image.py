import numpy as np
import settings
from AdvLaneline_utils import *
from AdvLanelinefit import *

# Define Complete Image Processing Pipeline

# @function name:           process_image
# @function description:    Define Complete Image Processing Pipeline
def process_image(img):
    global l_bestfit
    global r_bestfit
    global l_lane_inds_bestfit
    global r_lane_inds_bestfit   
    global l_fit
    global r_fit
    global l_lane_inds
    global r_lane_inds
    global stored_img
    new_img = np.copy(img)
    img_bin, Minv = image_pipeline("HLSCombine", settings.mtx, settings.dist, 0, os.path.join(os.path.abspath(os.getcwd()), 'output_images'), new_img, os.path.join(os.path.abspath(os.getcwd()), 'debug_folder'))

    l_fit, r_fit, l_lane_inds, r_lane_inds, _ = advlaneline_sliding_window_polyfit(img_bin)

    if l_fit is not None and r_fit is not None:
        # calculate x-intercept (bottom of image, x=image_height) for fits
        h = img.shape[0]
        l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
        r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
        x_int_diff = abs(r_fit_x_int-l_fit_x_int)
        if abs(350 - x_int_diff) > 100:
            l_fit = None
            r_fit = None

    #stored for best fit
    if l_fit is not None and r_fit is not None:    
        l_bestfit = l_fit
        r_bestfit = r_fit
        l_lane_inds_bestfit = l_lane_inds
        r_lane_inds_bestfit = r_lane_inds
        stored_img = img_bin

   
    # draw the current best fit if it exists
    if l_bestfit is not None and r_bestfit is not None:
        rad_l, rad_r, d_center = measure_curvature_distance(stored_img, l_bestfit, r_bestfit, l_lane_inds_bestfit, r_lane_inds_bestfit)
        img_out = laneline_plot(new_img, img_bin, l_bestfit, r_bestfit, Minv, (rad_l+rad_r)/2, d_center)
    else:
        img_out = new_img
        
    return img_out
