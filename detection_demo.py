# ================================================================
#
#   File name   : detection_demo.py
#   Author      : PyLessons
#   Created date: 2020-09-27
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : object detection image and video example
#
# ===============object deteciton=================================================
# from flask.app import Flask
from yolov3.configs import *
from yolov3.utils import (
    detect_image,
    detect_realtime,
    detect_video,
    Load_Yolo_model,
    detect_video_realtime_mp,
)
import tensorflow as tf
import numpy as np
import cv2
import os
import json

# ----------------flask--------------------------------------------------------------
# from flask import request
def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print ("---  new folder...  ---")
		print ("---  OK  ---")
 
	else:
		print ("---  There is this folder!  ---")

# -----------------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
yolo = Load_Yolo_model()
print("load_mode_finished")
for filename in os.listdir(r"./all_images"):
    # print(filename)
    
    image_path = "./all_images/" + filename
# video_path = "./IMAGES/test_recording_button.mp4"
    print(image_path)
    original_image, bboxes, cococlass = detect_image(
        yolo,
        image_path,
        "./IMAGES/street_after.jpg",
        input_size=YOLO_INPUT_SIZE,
        show=False,
        rectangle_colors=(255, 0, 0),
    )
    filename = os.path.splitext(filename)[0]
    path_fold = "./all_images_result/"+ filename 
    mkdir(path_fold)
    original_image_name = path_fold + "/original_image.npy"
    bboxes_name =  path_fold + "/bboxes.npy"    
    np.save(original_image_name, original_image)  # save
    np.save(bboxes_name, bboxes)  # save
    print("save_result"+original_image_name)
# np.save("cococlass.npy", cococlass)  # save/   "model_data/coco/coco.names"
# detect_video(
#     yolo,
#     video_path,
#     "./IMAGES/test_recording_button_after.mp4",
#     input_size=YOLO_INPUT_SIZE,
#     show=False,
#     rectangle_colors=(255, 0, 0),
# )
# detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255, 0, 0))

# detect_video_realtime_mp(video_path, "Output.mp4", input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0), realtime=False)
# app =Flask(__name__)

