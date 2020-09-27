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
# pip install pandas

import onnxruntime as rt
import numpy as np
from glob import glob
import os
import cv2
import time
#from mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
import pandas as pd
from tracker.mot import (tracking_table_init,
                         do_pairing,
                         remove_low_confidence,
                         none_type_checking)
from postgresql import (get_connection,
                        insert_age_gender,
                        disconnect)

cls2age = {0: '<10', 1: '10+', 2: '20+', 3: '30+', 4: '40+', 5: '50+', 6: '60+', 7: '>70'}
cls2gender = {0: 'f', 1: 'm'}

#
# model for face detection
#
# use yolo4 tiny to replace mtcnn
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

#
# model for age and gender inference
#
model_folder_path = ''
# img_folder_path = 'drive/My Drive/Tibame_AIoT_Project/Datasets/cleandataset'
# test_img_path = 'drive/My Drive/Tibame_AIoT_Project/test'
IMG_SIZE = 224
#onnx_model = "../vgg16_128.onnx"
onnx_model = "23-4-resnet_mlp512-128_+600.onnx"

sess = rt.InferenceSession(os.path.join(model_folder_path, onnx_model))
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

#
# tracking and counting
#
frame0_flag = 0  # 代表目前為最初始狀態，從frame_0開始
frame_count = 0
DB_W_FRAME = 60  # number of frame to write Database
DB_W_SEC = 10
# Init the DataFrame to record age and gender
df_age_gender = pd.DataFrame(columns=["ps", "date_created", "age" , "gender"])
print("Init df_age_gender:", df_age_gender)
# age_gender_count = np.zeros((8,2))
# df_age_gender_count = pd.DataFrame(age_gender_count, index=cls2age.values(), columns=cls2gender.values())
# print(df_age_gender_count)

#
# connect to postgreSQL database
#
cur, conn = get_connection()

all_start_time = time.time()
start_time = 0
time_01 = 0
time_02 = 0
time_03 = 0
time_04 = 0
end_time = 0
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
    global time_01, time_02, time_03, start_time, end_time
 
    faces, raw_results = detect_faces(image)
    time_01 = time.time()
    #print("detect_faces (yolo4 tiny) time: {}".format( time_01 - start_time ))      
    if len(faces) == 0:
        #print('No face', end="  ")
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

    time_02 = time.time()
    #print("preprocess time: {}".format( time_02 - time_01 )) 
    return prepro_faces, raw_results

#
# 用opencv讀取影片並做inference
#
# p1 = cv2.VideoCapture("a-mei.mp4")
p1 = cv2.VideoCapture("../TheHungerGames.mp4")
#p1 = cv2.VideoCapture("../ShootingApple.mp4")
# p1 = cv2.VideoCapture("../jin-ma-53.mp4")
#p1 = cv2.VideoCapture("../test_day_4.mp4")
#p1 = cv2.VideoCapture("video.mp4")
#p1 = cv2.VideoCapture(0)

print("高:", p1.get(4))
print("寬:", p1.get(3))
print("總影格:", p1.get(7))
#print("FPS:", p1.get(5))

#start_frame = 400
start_frame = 1160
#start_frame = 2560
#start_frame = 100 #930 #700 #280 #100
#start_frame = 0

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
        #
        # Inference, 第一個輸出是 probability of age,第二個輸出是 probability of gender
        # 
        x_input = np.array(prepro_faces)
        pred = sess.run([sess.get_outputs()[0].name,sess.get_outputs()[1].name], {input_name: x_input.astype('float32')})
        time_04 = time.time()
        #print("inference time({}): {}".format( onnx_model.split('_')[0], time_04 - time_03 ))

        # 
        # obj tracking
        # 
        if frame0_flag == 0:
            # INITIALIZATION
            new = tracking_table_init(raw_results, pred, frame0_flag)
            frame0_flag = 1
        else:
            old = new
            new = tracking_table_init(raw_results, pred, frame0_flag)
            #print("tracking_table:\n", new)

            # TRACKING
            do_pairing(new, old)  # pairing
            df_age_gender = remove_low_confidence(new, df_age_gender)  # removing
            none_type_checking(new)  # checking

            # if frame_count > DB_W_FRAME:
            #     print("df_age_gender:{}".format(df_age_gender['age'].count()), df_age_gender)

            # print("old:", len(old))
            # for i in range(len(old)):
            #     print("old_{}:{}".format(i, old[i]))
            # print("new:", len(new))    
            # for i in range(len(new)):
            #     print("new_{}:{}".format(i, new[i]))

        #標出預測結果
        for i in range(len(prepro_faces)):
            pred_age = np.array(pred[0][i]).argmax(axis=-1)
            pred_gender = np.array(pred[1][i]).argmax(axis=-1)

            #x1, y1, width, height = raw_results[i]['box']  #for MTCNN
            x1, y1, width, height = raw_results[i]
            cv2.rectangle(frame, (x1,y1), (x1+width, y1+height), (0,0,255), 2)
            #text = "{},{}  {}x{}".format(cls2age[pred_age], cls2gender[pred_gender], width, height)
            text = "{},{}".format(cls2age[pred_age], cls2gender[pred_gender])
            cv2.putText(frame, text, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # 標上ID
        for item in new:
            # Draw tracking IDs
            text = "ID={}, C={}" .format(item['id'], round(item['confidence'], 1))
            if item['confidence'] == 1:
                color = (0, 255, 0)
                cv2.putText(frame, text, (item['pos'][0] - 25, item['pos'][1] + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                color = (0, 255, 255)
                cv2.putText(frame, text, (item['pos'][0], item['pos'][1]), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


        end_time = time.time()
        fps_text = "FPS: %.2f" % (1 / (end_time - start_time))
        cv2.putText(frame, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # print("opencv FPS: {}".format( 1 / (end_time - time_04) ))
        #print("FPS: %.2f" % (1 / (end_time - start_time)))
        # print("predict age:",pred_age)
        # print("predict gender:",pred_gender)  
        cv2.imshow("onnx", frame)
        if cv2.waitKey(33) != -1: #按任意鍵離開(若沒按鍵則 cv2.waitKey() 會回傳-1)
            break
    else:
        break

    # 每 DB_W_FRAME 個 frame 把 DataFrame  df_age_gender 的資料寫入資料庫
    # if frame_count % DB_W_FRAME == 0 and frame_count != 0:
    #     print("frame{}\n".format(frame_count))
    # 每隔 秒把 DataFrame  df_age_gender 的資料寫入資料庫
    if (int(time.time() - all_start_time) % DB_W_SEC == 0) & (int(time.time() - all_start_time) > DB_W_SEC):
        print("time: {}\n".format(time.time() - all_start_time))
        # insert data to database
        for i in range(df_age_gender['age'].count()):
            #print("df_age_gender row{}".format(i), '\n', df_age_gender.iloc[i])
            row = df_age_gender.iloc[i]
            ps = row["ps"]
            date_created = time.asctime( time.localtime(time.time()) )  
            age = row["age"]
            gender = row["gender"]
             
            print("insert: ps={} date_created={} age={} gender={}".format(ps, date_created, age, gender))

            #insert_age_gender(cur, conn, date_created, age, gender, ps)



        # reinit df_age_gender DataFrame 
        df_age_gender = pd.DataFrame(columns=["ps", "date_created", "age" , "gender"])
        print("ReInit df_age_gender:", df_age_gender)
        # age_gender_count = np.zeros((8,2))
        # df_age_gender_count = pd.DataFrame(age_gender_count, 
        #     index=cls2age.values(), columns=cls2gender.values())
    frame_count += 1

# write the rest of data to DB
if df_age_gender['age'].count() != 0:
    for i in range(df_age_gender['age'].count()):
        #print("df_age_gender row{}".format(i), '\n', df_age_gender.iloc[i])
        row = df_age_gender.iloc[i]
        ps = row["ps"]
        date_created = time.asctime( time.localtime(time.time()) )  
        age = row["age"]
        gender = row["gender"]
            
        print("write the rest of data to DB: insert: ps={} date_created={} age={} gender={}".format(ps, date_created, age, gender))

        #insert_age_gender(cur, conn, date_created, age, gender, ps)


# disconnect with DB
disconnect(cur, conn)

p1.release()		
cv2.destroyAllWindows()