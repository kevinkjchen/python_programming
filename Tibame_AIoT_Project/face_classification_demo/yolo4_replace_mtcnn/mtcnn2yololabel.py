import numpy as np
from glob import glob
import os
import cv2
import time
from mtcnn import MTCNN
import shutil


#src_path = input("輸入原始路徑:") + '/'
#des_path = input("輸入目標路徑:") + '/'
src_path = 'test_0'
des_path = 'test_1'
img_folder_path = ""

def convert_bndbox2yololabel(img,x,y,w,h):
    img_h = img.shape[0]
    img_w = img.shape[1]    

    x_center = (x + w/2.)/img_w
    y_center = (y + h/2.)/img_h
    w_yolo = w/img_w
    h_yolo = h/img_h

    return (x_center, y_center, w_yolo, h_yolo)

detector = MTCNN() 
print("detector:", type(detector))
paths = glob(src_path+"/*.jpg")

for i, path in enumerate(paths):
    #print("idx:", i, "cls:", y_cls[i], path)

    # 讀取圖片,切下臉的部分,並使用借來的模型的預處理方式來作預處理           
    img = cv2.imread(os.path.join(img_folder_path,path))[:,:,::-1]
    # 取得 bounding boxes
    results = detector.detect_faces(img)
    if len(results) == 0:
        continue
    # 取得第一張臉的 bounding box, and then convert to yolo's label
    x1, y1, width, height = results[0]['box']           
    yolo_label = convert_bndbox2yololabel(img, x1, y1, width, height)

    # 檔名
    #fn = path.split('.')[0].split('/')[-1]	# for linux
    fn = path.split('.')[0].split('\\')[-1]	# for windows
    # 複製原始圖片到 des_path 
    shutil.copyfile( os.path.join(src_path, fn +'.jpg'), os.path.join(des_path, fn +'.jpg') )
    # 在 des_path 建立圖片對應的 txt file
    with open(os.path.join(des_path, fn + '.txt'), "w", encoding="utf-8") as f:
        f.write("0 {} {} {} {}".format(yolo_label[0], yolo_label[1], yolo_label[2], yolo_label[3]))


   