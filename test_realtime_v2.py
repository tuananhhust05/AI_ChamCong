
import os
#from chamcong365_finnal.src.facenet import load_model
import cv2
import numpy as np
import argparse
import warnings  # get warnings
import time
import sys
from Face_Anti_Spoofing.src_anti.anti_spoof_predict import AntiSpoofPredict
from Face_Anti_Spoofing.src_anti.generate_patches import CropImage
from Face_Anti_Spoofing.src_anti.utility import parse_model_name

# from src_anti.anti_spoof_predict import AntiSpoofPredict
# from src_anti.generate_patches import CropImage
# from src_anti.utility import parse_model_name

warnings.filterwarnings('ignore')
sys.path.append('../')
# time_load_model = 0

model_test = AntiSpoofPredict(0)

# image = './Face_Anti_Spoofing/tes.jpg'
# model_dir='./resources/anti_spoof_models/'
# device_id = 0
model_name ='./Face_Anti_Spoofing/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth'
model_test._load_model(model_name)

def detect_spoofing(image):
    image_cropper = CropImage()
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    h_input, w_input, model_type, scale = parse_model_name('4_0_0_80x80_MiniFASNetV1SE.pth')
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
    #cv2.imwrite('img.png',img)
    prediction = model_test.predict(img)
    
    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    return label == 1



