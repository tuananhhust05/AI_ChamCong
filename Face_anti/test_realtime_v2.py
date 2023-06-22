# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings  # get warnings
import time
import sys
# from Face_Anti_Spoofing.src_anti.anti_spoof_predict import AntiSpoofPredict
# from Face_Anti_Spoofing.src_anti.generate_patches import CropImage
# from Face_Anti_Spoofing.src_anti.utility import parse_model_name

from src_anti.anti_spoof_predict import AntiSpoofPredict
from src_anti.generate_patches import CropImage
from src_anti.utility import parse_model_name

warnings.filterwarnings('ignore')
sys.path.append('../')
time_load_model = 0

model_test = AntiSpoofPredict(0)

# image = './Face_Anti_Spoofing/tes.jpg'
# model_dir='./resources/anti_spoof_models/'
# device_id = 0

def detect_spoofing(image, model_dir='./resources/anti_spoof_models/'):
# def detect_spoofing(image):


    # global model_test
    time_c = time.time()
    image_cropper = CropImage()

    image_bbox = model_test.get_bbox(image)
    # print(image_bbox)
    time_load_model = time.time() - time_c
    print("time_detect_face", time_load_model)
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result


    time_model = 0
    # a_a = time.time()
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False

    img = image_cropper.crop(**param)



    start = time.time()
    prediction = model_test.predict(img, os.path.join(model_dir, model_name))
    test_speed = time.time() - start
    print("Time_detect:", test_speed)
    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    return label == 1



