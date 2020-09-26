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

#import onnxruntime as rt
import onnx
import onnx_tensorrt.backend as backend

import numpy as np
from glob import glob
import os
import cv2
import time

from mtcnn import MTCNN
from keras_vggface.utils import preprocess_input

cls2age = {0: '<10', 1: '10+', 2: '20+', 3: '30+', 4: '40+', 5: '50+', 6: '60+', 7: '>70'}
cls2gender = {0: 'f', 1: 'm'}

model_folder_path = ''
# img_folder_path = 'drive/My Drive/Tibame_AIoT_Project/Datasets/cleandataset'
# test_img_path = 'drive/My Drive/Tibame_AIoT_Project/test'
IMG_SIZE = 224

detector = MTCNN()

ONNX_TENSORRT = 1
if ONNX_TENSORRT: 
    model = onnx.load( os.path.join(model_folder_path,"resnet_512-128.onnx") )
    engine = backend.prepare(model, device='CUDA:1')
else:    
    sess = rt.InferenceSession(os.path.join(model_folder_path,"resnet_512-128.onnx"))
    input_name = sess.get_inputs()[0].name
    print("input name", input_name)
    input_shape = sess.get_inputs()[0].shape
    print("input shape", input_shape)
    input_type = sess.get_inputs()[0].type
    print("input type", input_type)
    output = sess.get_outputs()
    print("output name:", output[0].name, output[1].name)
    print("output shape", output[0].shape, output[1].shape)
    print("output type", output[0].type, output[1].type)



start_time = 0
time_01 = 0
time_02 = 0
time_03 = 0
time_04 = 0
end_time = 0
# detect face
def detect_faces(img):
    face_imgs = []

    results = detector.detect_faces(img)
    # extract the bounding box from the first face
    # print('# of faces: ', len(results))
    if len(results) == 0:
        #print(face_imgs, results)
        return face_imgs, results
    for i in range(len(results)):
        x1, y1, width, height = results[i]['box']
        x2, y2 = x1 + width, y1 + height
        patch = img[y1:y2, x1:x2] # crop face
        face_imgs.append(patch)
     
    return face_imgs, results

# 每張圖片可偵測多個人臉,切下臉的部分,並使用借來的模型的預處理方式來作預處理
def preprocess_image(image):
    global time_01, time_02, time_03, start_time, end_time
 
    faces, raw_results = detect_faces(image)
    time_01 = time.time()
    print("mtcnn time: {}".format( time_01 - start_time ))      
    if len(faces) == 0:
        print('No face', end="  ")
        return [], []
    # print(faces[0].shape) 

    prepro_faces = []
    print("faces:",len(faces), end="  ")
    for face in faces:
        if face.shape[0] == 0 or face.shape[1] == 0:
            continue
        crop_face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        
        # 使用借來的模型的預處理方式來作預處理
        prepro_face = preprocess_input(np.array(crop_face, dtype=float)) 
        
        prepro_faces.append(prepro_face)

    time_02 = time.time()
    print("preprocess time: {}".format( time_02 - time_01 )) 
    return prepro_faces, raw_results

#
# 用opencv讀取影片並做inference
#
# p1 = cv2.VideoCapture("a-mei.mp4")
p1 = cv2.VideoCapture("TheHungerGames.mp4")
# p1 = cv2.VideoCapture("ShootingApple.mp4")
# p1 = cv2.VideoCapture("jin-ma-53.mp4")
print("高:", p1.get(4))
print("寬:", p1.get(3))
print("總影格:", p1.get(7))
print("FPS:", p1.get(5))

#start_frame = 400
start_frame = 950
#start_frame = 2560
#start_frame = 100 #930 #700 #280 #100

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
        start_time = time.time()
        frame = cv2.resize(frame_ori, (960, 540))
        #frame = frame_ori
        #偵測一個frame中的多張臉, 切下臉的部分, 並做預處理
        prepro_faces, raw_results = preprocess_image(frame)
        
        #print(prepro_faces, raw_results, len(prepro_faces), len(raw_results))
        if len(prepro_faces) == 0:
            continue
            
        time_03 = time.time()    
        x_input = np.array(prepro_faces)
        #第一個輸出是age,第二個輸出是gender
        if ONNX_TENSORRT:
            pred = engine.run(x_input)[0]
        else:
            pred = sess.run([sess.get_outputs()[0].name,sess.get_outputs()[1].name], {input_name: x_input.astype('float32')})
        time_04 = time.time()
        print("inference time: {}".format( time_04 - time_03 ))

        #標出預測結果
        for i in range(len(prepro_faces)):
            pred_age = np.array(pred[0][i]).argmax(axis=-1)
            pred_gender = np.array(pred[1][i]).argmax(axis=-1)

            x1, y1, width, height = raw_results[i]['box']
            cv2.rectangle(frame, (x1,y1), (x1+width, y1+height), (0,0,255), 2)
            text = "{},{}".format(cls2age[pred_age], cls2gender[pred_gender])
            cv2.putText(frame, text, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        end_time = time.time()
        fps_text = "FPS: %.2f" % (1 / (end_time - start_time))
        cv2.putText(frame, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        # print("opencv FPS: {}".format( 1 / (end_time - time_04) ))
        # print("FPS: %.2f" % (1 / (end_time - start_time)))
        # print("predict age:",pred_age)
        # print("predict gender:",pred_gender)  
        cv2.imshow("onnx", frame)
        if cv2.waitKey(33) != -1: #按任意鍵離開(若沒按鍵則 cv2.waitKey() 會回傳-1)
            break
    else:
        break
p1.release()		
cv2.destroyAllWindows()