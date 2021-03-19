import sys
import os
import json
import glob


import numpy as np
import dlib
import cv2
from PIL import Image
import time
from tqdm import tqdm

if len(sys.argv) != 2:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

#첫번째 매개변수로 68개의 얼굴 랜드마크 학습된 모델 데이터
predictor_path = sys.argv[1]
# 랜드마크를 적용할 이미지들을 모다운 폴더
#faces_folder_path = sys.argv[2]


ALL = list(range(0,68))

detector = dlib.get_frontal_face_detector() #얼굴 인식용 클래스 생성
predictor = dlib.shape_predictor(predictor_path) # 인식된 얼굴에서 랜드마크 찾는 클래스 생성
win = dlib.image_window()

root = '/home/ubuntu01/Multimodal_yj/Wav2Lip/gini'
win = dlib.image_window()
for i, f in enumerate(glob.glob(os.path.join(root, "*.jpg"))): #해당 폴더를 모두 돌아가며 jpg 파일을 찾음
    img = cv2.imread(f)
    img = dlib.load_rgb_image(f)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


    win.clear_overlay()
    win.set_image(img)

    dets = detector(img, 1)
    h = np.array(img).shape[0]
    w = np.array(img).shape[1]
    print(h, w)
    print(dets)
    tmp = dlib.rectangle(0,0,w,h)
    
    cv2.rectangle(img, (tmp.left(),tmp.top()), (tmp.right(),tmp.bottom()), (0,255,0),2)

    shape = predictor(img, tmp) #dets[0]
    win.add_overlay(shape)
    win.add_overlay(tmp)

    for j in range(68):
        x, y = shape.part(j).x, shape.part(j).y
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

    
    cv2.imwrite('output_{}.jpg'.format(i), img)
    

    dlib.hit_enter_to_continue()
    


