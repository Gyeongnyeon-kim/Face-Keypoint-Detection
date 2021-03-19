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
#win = dlib.image_window()

root = '/home/ubuntu01/Multimodal_yj/Wav2Lip/lrs2_preprocessed'
load_landmark = np.load(os.path.join(root, 'Face_Landmarks.npz'))

folders = os.listdir(root) #55434434333 ....
empty_list = []
Landmark_Lists = []
Img_size = []
Loc = []
for folder_ in tqdm(folders):
    root2 = os.path.join(root, folder_)
    folders_2 = os.listdir(root2) #00001 00002 00003 ....
    for folder in tqdm(folders_2):
        root3 = os.path.join(root2, folder)
        time.sleep(0.1)
        # pbar.set_description(f'Processing {folder_}')
        #empty_list.append(folder_)
        for i, f in enumerate(glob.glob(os.path.join(root3, "*.jpg"))): #해당 폴더를 모두 돌아가며 jpg 파일을 찾음
            rest_loc = f.split("lrs2_preprocessed")[1]
            Loc.append(rest_loc)
            #print("Processing file: {}".format(f))
            img = dlib.load_rgb_image(f) #파일에서 이미지 불러오기
            #img = cv2.resize(img, dsize = (100,100),interpolation=cv2.INTER_AREA)
            #win.clear_overlay()
            #win.set_image(img)

            dets = detector(img, 1)
            h = np.array(img).shape[0]
            w = np.array(img).shape[1]
            tmp = dlib.rectangle(0,0,h,w)
            Img_size.append(tmp)
            #print(tmp)

            landmarks = predictor(img, tmp)

            landmark_list = []
            for p in landmarks.parts():
                landmark_list.append([p.x, p.y])   
            #print("Part 0: {}, Part 1: {}, Part 2: {} ...".format(landmarks.part(0),
            #                                        landmarks.part(1),
            #                                        landmarks.part(2)))
            Landmark_Lists.append(landmark_list)
    print("{} Saved...".format(folder_))

Landmark_Lists = np.array(Landmark_Lists)
Img_size = np.array(Img_size)
Loc = np.array(Loc)
np.savez(os.path.join(root,'Face_Landmarks'),Landmark_Lists=Landmark_Lists , Img_size= Img_size, Loc = Loc)
                # Draw the face landmarks on the screen.
                #win.add_overlay(shape)
            
            #win.add_overlay(dets)
        #dlib.hit_enter_to_continue()
