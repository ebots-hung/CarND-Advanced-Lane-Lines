import numpy as np
import settings
from AdvLaneline_utils import *
from AdvLanelinefit import *

# Define Complete Image Processing Pipeline

# @function name:           process_image
# @function description:    Define Complete Image Processing Pipeline
def process_image(img):
    global l_line
    global r_line
    new_img = np.copy(img)
    img_bin, Minv = image_pipeline("HLSCombine", settings.mtx, settings.dist, 0, os.path.join(os.path.abspath(os.getcwd()), 'output_images'), new_img, os.path.join(os.path.abspath(os.getcwd()), 'debug_folder'))
    l_line = AdvLanelinefit()
    r_line = AdvLanelinefit()
    # if both left and right lines were detected last frame, use polyfit_using_prev_fit, otherwise use sliding window
    # if not l_line.detected or not r_line.detected:
    l_fit, r_fit, l_lane_inds, r_lane_inds, _ = advlaneline_sliding_window_polyfit(img_bin)
    # else:
    #     l_fit, r_fit, l_lane_inds, r_lane_inds = advlaneline_polyfit_using_prev_fit(img_bin, l_line.best_fit, r_line.best_fit)
        
    # invalidate both fits if the difference in their x-intercepts isn't around 350 px (+/- 100 px)
    # if l_fit is not None and r_fit is not None:
    #     # calculate x-intercept (bottom of image, x=image_height) for fits
    #     h = img.shape[0]
    #     l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
    #     r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
    #     x_int_diff = abs(r_fit_x_int-l_fit_x_int)
    #     if abs(350 - x_int_diff) > 100:
    #         l_fit = None
    #         r_fit = None
            
    # l_line.add_fit(l_fit, l_lane_inds)
    # r_line.add_fit(r_fit, r_lane_inds)
    
    # draw the current best fit if it exists
    # if l_line.best_fit is not None and r_line.best_fit is not None:
    rad_l, rad_r, d_center = measure_curvature_distance(img_bin, l_fit, r_fit, l_lane_inds, r_lane_inds)
    img_out = laneline_plot(new_img, img_bin, l_fit, r_fit, Minv, (rad_l+rad_r)/2, d_center)
    # else:
    #     img_out = new_img
        
    return img_out
