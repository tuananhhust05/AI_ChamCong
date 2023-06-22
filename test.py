# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
# import argparse
import warnings
import time
# import imutils

from imutils.video import VideoStream
from src.src_anti.anti_spoof_predict import AntiSpoofPredict
from src.src_anti.generate_patches import CropImage
from src.src_anti.utility import parse_model_name

warnings.filterwarnings('ignore')

cap = VideoStream().start()


# SAMPLE_IMAGE_PATH = "./images/sample/"
# SAMPLE_IMAGE_PATH_RESULT = "./images/result/"


# Check image input
# def check_image(image):
#     height, width, channel = image.shape
#     if width/height != 3/4:
#         print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
#         return False
#     else:
#         return True

# class faceAnti:
def test():
    while True:
        frame = cap.read()
        frame = cv2.flip(frame, 1)
        device_id = 0
        model_dir = './resources/anti_spoof_models/'
        # frame = '3.png'
        model_test = AntiSpoofPredict(device_id)
        image_cropper = CropImage()
        # frame = cv2.imread(SAMPLE_IMAGE_PATH + frame)
        # result = check_image(frame)
        # if result is False:
        #     return
        image_bbox = model_test.get_bbox(frame)
        prediction = np.zeros((1, 3))
        test_speed = 0
        # sum the prediction from single model's result
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": frame,
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
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))
            test_speed += time.time() - start

        # draw result of prediction
        label = np.argmax(prediction)
        value = prediction[0][label] / 2

        # if label == 1:
        #     print("Image is Real Face. Accuracy: {:.2f}.".format(value))
        #     result_text = "RealFace Score: {:.2f}".format(value)
        #     color = (255, 0, 0)
        # else:
        #     print("Image is Fake Face. Accuracy: {:.2f}.".format(value))
        #     result_text = "FakeFace Score: {:.2f}".format(value)
        #     color = (0, 0, 255)

        if label == 1:
            print("Image is Real Face. Accuracy: {:.2f}.".format(value))
            result_text = "RealFace Score: {:.2f}".format(value)
            color = (255, 0, 0)
        else:
            print("Image is Fake Face. Accuracy: {:.2f}.".format(value))
            result_text = "FakeFace Score: {:.2f}".format(value)
            color = (0, 0, 255)
        print("Prediction cost {:.2f} s".format(test_speed))
        cv2.rectangle(
            frame,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 2)
        cv2.putText(
            frame,
            result_text,
            (image_bbox[0], image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX, 0.5 * frame.shape[0] / 1024, color)

        # format_ = os.path.splitext(frame)[-1]
        # result_image_name = frame.replace(format_, "_result" + format_)
        # cv2.imwrite(SAMPLE_IMAGE_PATH_RESULT + result_image_name, frame)

    # if __name__ == "__main__":
    #     desc = "test"
    #     parser = argparse.ArgumentParser(description=desc)
    #     parser.add_argument(
    #         "--device_id",
    #         type=int,
    #         default=0,
    #         help="which gpu id, [0/1/2/3]")
    #     parser.add_argument(
    #         "--model_dir",
    #         type=str,
    #         default="./resources/anti_spoof_models",
    #         help="model_lib used to test")
    #     parser.add_argument(
    #         "--image_name",
    #         type=str,
    #         default="",
    #         help="image used to test")
    #     args = parser.parse_args()
    #     test()

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test()
