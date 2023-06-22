import os
import numpy as np
import cv2
import warnings  # get warnings
import time
import sys
from Face_Anti_Spoofing.src_anti.anti_spoof_predict_v2 import AntiSpoofPredict
from Face_Anti_Spoofing.src_anti.generate_patches import CropImage
from Face_Anti_Spoofing.src_anti.utility import parse_model_name

warnings.filterwarnings('ignore')
sys.path.append('../../')
time_load_model = 0
model_test = {}

# 2 model để nhận biết ảnh giả mạo và không giả mạo ở 2 kích thước khác nhau
model_test[0] = AntiSpoofPredict(0)
model_test[0]._load_model('./Face_Anti_Spoofing/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth')

model_test[1] = AntiSpoofPredict(0)
model_test[1]._load_model('./Face_Anti_Spoofing/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth')


def detect_spoofing(image, model_dir='./Face_Anti_Spoofing/resources/anti_spoof_models/',high_acc = True):
    # Phát hiện khuôn mặt giả mạo
    global model_test
    image_cropper = CropImage()
    labels = []
    imgs = None
    for i,model_name in enumerate(os.listdir(model_dir)):
        prediction = np.zeros((1, 3))
        bboxes = model_test[i].get_bbox(image)
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bboxes": bboxes,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        imgs = image_cropper.crop(**param)
    
    for img in imgs:
        prediction = np.zeros((1, 3))
        max_predict = []
        for i,model_name in enumerate(os.listdir(model_dir)):
            predict_spoof = model_test[i].predict(img, os.path.join(model_dir, model_name))
            prediction += predict_spoof
            predict_spoof = np.where(predict_spoof == np.max(predict_spoof), 1, 0)
            max_predict.append(predict_spoof)
        
        if (max_predict[0] == max_predict[1]).all():
            labels.append(1)
        else:
            labels.append(0)
            
            
    if len(set(labels)) == 1 and 1 in set(labels):
        return True
    else:
        return False
        
    
    