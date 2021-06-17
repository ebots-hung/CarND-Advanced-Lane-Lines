import numpy as np
import settings
from AdvLaneline_utils import *
from settings import *

# @function name:           process_image
# @function description:    Define Complete Image Processing Pipeline
def process_image(img):
    
    new_img = np.copy(img)
    img_bin, Minv = image_pipeline("HLSCombine", settings.mtx, settings.dist, 0, os.path.join(os.path.abspath(os.getcwd()), 'output_images'), new_img, os.path.join(os.path.abspath(os.getcwd()), 'debug_folder'))
    #stored for best fit
    if settings.l_fit is not None and settings.r_fit is not None:    
        settings.l_fit, settings.r_fit, settings.l_lane_inds, settings.r_lane_inds = advlaneline_polyfit_using_prev_fit(img_bin, settings.l_bestfit, settings.r_bestfit)
    else: 
        settings.l_fit, settings.r_fit, settings.l_lane_inds, settings.r_lane_inds, _ = advlaneline_sliding_window_polyfit(img_bin)
    
    if settings.l_fit is not None and settings.r_fit is not None:
        # calculate x-intercept (bottom of image, x=image_height) for fits
        h = img.shape[0]
        l_fit_x_int = settings.l_fit[0]*h**2 + settings.l_fit[1]*h + settings.l_fit[2]
        r_fit_x_int = settings.r_fit[0]*h**2 + settings.r_fit[1]*h + settings.r_fit[2]
        x_int_diff = abs(r_fit_x_int-l_fit_x_int)
        if abs(350 - x_int_diff) > 100:
            settings.l_fit = None
            settings.r_fit = None

    #stored for best fit
    if settings.l_fit is not None and settings.r_fit is not None:    
        settings.l_bestfit = settings.l_fit
        settings.r_bestfit = settings.r_fit
        settings.l_lane_inds_bestfit = settings.l_lane_inds
        settings.r_lane_inds_bestfit = settings.r_lane_inds
        settings.stored_img = img_bin

    # draw the current best fit if it exists
    if settings.l_bestfit is not None and settings.r_bestfit is not None:
        rad_l, rad_r, d_center = measure_curvature_distance(settings.stored_img, settings.l_bestfit, settings.r_bestfit, settings.l_lane_inds_bestfit, settings.r_lane_inds_bestfit)
        img_out = laneline_plot(new_img, img_bin, settings.l_bestfit, settings.r_bestfit, Minv, (rad_l+rad_r)/2, d_center)
    else:
        rad_l, rad_r, d_center = measure_curvature_distance(img_bin, settings.l_fit, settings.r_fit, settings.l_lane_inds, settings.r_lane_inds)
        img_out = laneline_plot(new_img, img_bin, settings.l_bestfit, settings.r_bestfit, Minv, (rad_l+rad_r)/2, d_center)
        
    return img_out
