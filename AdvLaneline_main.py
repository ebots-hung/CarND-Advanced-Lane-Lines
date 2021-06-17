# -*- coding: utf-8 -*-
# @Author:              QH Lam
# @Date:                2021-02-27
# @Module name:         utils.py
# @Module description:  Advanced Lane lines main function

import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import settings

from moviepy.editor import VideoFileClip
from AdvLaneline_utils import *
from AdvLaneline_process_image import *


dump_image_folderpath = "output_images"
test_image_folderpath = "test_images"

if __name__ == '__main__':
    currentfolderpath = os.path.abspath(os.getcwd())
    filetoread = os.path.join(currentfolderpath, dump_image_folderpath, "camcalibration.ini")
    settings.read_calibration_file(filetoread)
    imgcopy = cv2.imread(os.path.join(currentfolderpath, test_image_folderpath, "test1.jpg"))
    settings.init(imgcopy)
    video_input1 = VideoFileClip('project_video.mp4')#.subclip(22,26)
    video_output1 = 'project_video_output.mp4'
    processed_video = video_input1.fl_image(process_image)
    processed_video.write_videofile(video_output1, audio=False)