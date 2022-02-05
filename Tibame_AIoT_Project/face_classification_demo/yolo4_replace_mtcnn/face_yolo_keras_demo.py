#
# Use onnxruntime to predict
#
# need install:
# pip install tensorflow==2.2.0
# pip install opencv-python
# pip install onnxruntime
# pip install mtcnn
# pip install keras_vggface
# pip install keras_applications

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
# import onnxruntime as rt
import numpy as np
from glob import glob
import os
import cv2
import time

#from mtcnn import MTCNN
from keras_vggface.utils import preprocess_input


cls2age = {0: '<10', 1: '10+', 2: '20+', 3: '30+', 4: '40+', 5: '50+', 6: '60+', 7: '>70'}
cls2gender = {0: 'f', 1: 'm'}

model_folder_path = ''
# img_folder_path = 'drive/My Drive/Tibame_AIoT_Project/Datasets/cleandataset'
# test_img_path = 'drive/My Drive/Tibame_AIoT_Project/test'
IMG_SIZE = 224

#use yolo4 tiny to replace mtcnn
#detector = MTCNN()
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
net = cv2.dnn.readNetFromDarknet("yolov4-person_tiny.cfg","yolov4-person_tiny_last.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/256)    

# tf.keras
age_gender_model = load_model(os.path.join(model_folder_path,'../23-1_resnet_mlp512-128_bs64_save.h5'))

# # onnxruntime
# sess = rt.InferenceSession(os.path.join(model_folder_path,"resnet_512-128.onnx"))
# input_name = sess.get_inputs()[0].name
# print("input name", input_name)
# input_shape = sess.get_inputs()[0].shape
# print("input shape", input_shape)
# input_type = sess.get_inputs()[0].type
# print("input type", input_type)
# output = sess.get_outputs()
# print("output name:", output[0].name, output[1].name)
# print("output shape", output[0].shape, output[1].shape)
# print("output type", output[0].type, output[1].type)



# detect face
def detect_faces(img):
    face_imgs = []
    #
    # use MTCNN　
    #
    # results = detector.detect_faces(img)  

    #
    # use YOLO4 tiny 
    #
    classes, scores, boxes = model.detect(img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    results = boxes
    #print('# of faces: ', len(results), "scores:", scores)

    # extract the bounding box from the faces
    if len(results) == 0:
        #print(face_imgs, results)
        return face_imgs, results
    for i in range(len(results)):
        #x1, y1, width, height = results[i]['box']  #for MTCNN
        x1, y1, width, height = results[i]
        x2, y2 = x1 + width, y1 + height
        patch = img[y1:y2, x1:x2] # crop face
        face_imgs.append(patch)
     
    return face_imgs, results

# 每張圖片可偵測多個人臉,切下臉的部分,並使用借來的模型的預處理方式來作預處理
def preprocess_image(image):
    faces, raw_results = detect_faces(image)
    if len(faces) == 0:
        print('No face', end="  ")
        return [], []
    # print(faces[0].shape) 

    prepro_faces = []
    #print("faces:",len(faces), end="  ")
    for face in faces:
        if face.shape[0] == 0 or face.shape[1] == 0:
            continue
        crop_face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        
        # 使用借來的模型的預處理方式來作預處理
        prepro_face = preprocess_input(np.array(crop_face, dtype=float)) 
        
        prepro_faces.append(prepro_face)

    return prepro_faces, raw_results

#
# 用opencv讀取影片並做inference
#
# p1 = cv2.VideoCapture("a-mei.mp4")
p1 = cv2.VideoCapture("../TheHungerGames.mp4")
# p1 = cv2.VideoCapture("../ShootingApple.mp4")
# p1 = cv2.VideoCapture("../jin-ma-53.mp4")
print("高:", p1.get(4))
print("寬:", p1.get(3))
print("總影格:", p1.get(7))
print("FPS:", p1.get(5))

#start_frame = 400
start_frame = 950
#start_frame = 2560
#start_frame = 100 

end_frame = start_frame + 60.0
p1.set(1, start_frame) # set 當前影格
while p1.isOpened()==True:
    ret, frame_ori = p1.read()
    #p1.set(1, p1.get(1)+10.)
    if ret == True:
        frame_no = p1.get(1)
        print("影格:", frame_no, end=" ")
        if frame_no > end_frame:
            break
        start = time.time()
        frame = cv2.resize(frame_ori, (960, 540))
        #frame = frame_ori
        #偵測一個frame中的多張臉, 切下臉的部分, 並做預處理
        prepro_faces, raw_results = preprocess_image(frame)
        #print(prepro_faces, raw_results, len(prepro_faces), len(raw_results))
        if len(prepro_faces) == 0:
            continue

        #
        # Inference, 第一個輸出是age,第二個輸出是gender
        # 
        x_input = np.array(prepro_faces)
        pred = age_gender_model.predict(np.array(x_input))
        #pre[0] is predicted probabilities for age
        #pre[1] is predicted probabilities for gender
        #pred_age = pred[0].argmax(axis=-1)
        #pred_gender = pred[1].argmax(axis=-1)

        #標出預測結果
        for i in range(len(prepro_faces)):
            pred_age = np.array(pred[0][i]).argmax(axis=-1)
            pred_gender = np.array(pred[1][i]).argmax(axis=-1)

            #x1, y1, width, height = raw_results[i]['box']  #for MTCNN
            x1, y1, width, height = raw_results[i]
            cv2.rectangle(frame, (x1,y1), (x1+width, y1+height), (0,0,255), 2)
            text = "{},{}".format(cls2age[pred_age], cls2gender[pred_gender])
            cv2.putText(frame, text, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        end = time.time()
        fps_text = "FPS: %.2f" % (1 / (end - start))
        cv2.putText(frame, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        # print("FPS: %.2f" % (1 / (end - start)))
        # print("predict age:",pred_age)
        # print("predict gender:",pred_gender)  
        cv2.imshow("tf.keras", frame)
        if cv2.waitKey(33) != -1: #按任意鍵離開(若沒按鍵則 cv2.waitKey() 會回傳-1)
            break
    else:
        break
p1.release()		
cv2.destroyAllWindows()