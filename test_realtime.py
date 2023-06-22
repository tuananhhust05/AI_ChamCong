import os
import cv2
import numpy as np
import argparse
import warnings  # get warnings
import time
import sys
from Face_Anti_Spoofing.src_anti.anti_spoof_predict import AntiSpoofPredict
from Face_Anti_Spoofing.src_anti.generate_patches import CropImage
from Face_Anti_Spoofing.src_anti.utility import parse_model_name

warnings.filterwarnings('ignore')
sys.path.append('../')
model_test = AntiSpoofPredict(0)
def detect_spoofing(image, model_dir='./Face_Anti_Spoofing/resources/anti_spoof_models/'):
    global model_test
    image_cropper = CropImage()
    
    image_bbox = model_test.get_bbox(image)
    
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
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
        

        # start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
	# test_speed += time.time() - start

    # draw result of prediction
    # print(prediction)
    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    return label == 1
   
