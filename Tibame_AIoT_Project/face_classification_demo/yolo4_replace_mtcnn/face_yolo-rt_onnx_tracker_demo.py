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
import trt as trtpy
from trt import (   get_engine,
                        allocate_buffers,
                        detect,
                        do_inference,
                        engine_path
                    )


cls2age = {0: '<10', 1: '10+', 2: '20+', 3: '30+', 4: '40+', 5: '50+', 6: '60+', 7: '>70'}
cls2gender = {0: 'f', 1: 'm'}

#
# model for face detection
#
# use yolo4 tiny to replace mtcnn
#detector = MTCNN()
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

#class_names = []
#with open("classes.txt", "r") as f:
#    class_names = [cname.strip() for cname in f.readlines()]
#net = cv2.dnn.readNetFromDarknet("yolov4-person_tiny.cfg","yolov4-person_tiny_last.weights")
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
#model = cv2.dnn_DetectionModel(net)
#model.setInputParams(size=(416, 416), scale=1/256)    

#
# model for age and gender inference
#
model_folder_path = ''
# img_folder_path = 'drive/My Drive/Tibame_AIoT_Project/Datasets/cleandataset'
# test_img_path = 'drive/My Drive/Tibame_AIoT_Project/test'
IMG_SIZE = 224
#onnx_model = "../vgg16_128.onnx"
onnx_model = "Resnet_mlp512-128_bs64_23-3.onnx"

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
frame0_flag = 0  # start from frame_0
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
    results_boxes = []
    #
    # use YOLO4 tiny 
    #
    # classes, scores, boxes = model.detect(img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # results = boxes
    #print('# of faces: ', len(results), "scores:", scores)
    img_height, img_width = img.shape[:2]

    # use TensorRT as finference engine ace_detect_yolo model 
    print(context, buffers)
    #boxes = detect(context, buffers, img, (IMG_SIZE, IMG_SIZE), 1)
    boxes = detect(context, buffers, img, (416, 416), 1)
    
    print(type(boxes[0]), len(boxes[0]), boxes[0])
    # extract the bounding box from the faces
    if len(boxes[0]) == 0:
        #print(face_imgs, results)
        return face_imgs, boxes[0]

    for box in boxes[0]:
        x1 = int(box[0] * img_width)
        y1 = int(box[1] * img_height)
        x2 = int(box[2] * img_width)
        y2 = int(box[3] * img_height)

        patch = img[y1:y2, x1:x2] # crop face
        face_imgs.append(patch)   
        results_boxes.append([x1, y1, x2-x1, y2-y1])  
        print(x1, y1, x2-x1, y2-y1)  

    # for i in range(len(results)):
    #     #x1, y1, width, height = results[i]['box']  #for MTCNN
    #     print(i, results[i])
    #     x1, y1, width, height, _, _, _ = results[i]
    #     x2, y2 = x1 + width, y1 + height
    #     patch = img[y1:y2, x1:x2] # crop face
    #     face_imgs.append(patch)
    
    return face_imgs, results_boxes

# detect faces and preprocess
def preprocess_image(image):
    global time_01, time_02, time_03, start_time, end_time
 
    faces, raw_results = detect_faces(image)
    time_01 = time.time()
    print("detect_faces (yolo4 tiny) time: {}".format( time_01 - start_time ))      
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
        
        # preprocess
        prepro_face = preprocess_input(np.array(crop_face, dtype=float)) 
        
        prepro_faces.append(prepro_face)

    time_02 = time.time()
    #print("preprocess time: {}".format( time_02 - time_01 )) 
    return prepro_faces, raw_results

print("01")
#
# use opencv to read video
#
# p1 = cv2.VideoCapture("a-mei.mp4")
# p1 = cv2.VideoCapture("../TheHungerGames.mp4")
p1 = cv2.VideoCapture("ShootingApple.mp4")
# p1 = cv2.VideoCapture("../jin-ma-53.mp4")
#p1 = cv2.VideoCapture("../test_day_5.mp4")
#p1 = cv2.VideoCapture(0)
print("02")


#trt_context = 0
#trt_buffers = 0
with get_engine(engine_path) as engine, engine.create_execution_context() as context:
    buffers = allocate_buffers(engine, 1)
    print("context, buffers", type(context), type(buffers))
    print("get_binding_shape", engine.get_binding_shape(0))
    #context.set_binding_shape(0, (1, 3, IMG_SIZE, IMG_SIZE))
    context.set_binding_shape(0, (1, 3, 416, 416))
    #[TensorRT] ERROR: Parameter check failed at: engine.cpp::setBindingDimensions::1045, condition: profileMinDims.d[i] <= dimensions.d[i]

    #context.set_binding_shape(0, (3, IMG_SIZE, IMG_SIZE))
    #[TensorRT] ERROR: Parameter check failed at: engine.cpp::setBindingDimensions::1036, condition: engineDims.nbDims == dimensions.nbDims

    print("03")
    global trt_context, trt_buffers
    trt_context = context
    trt_buffers = buffers
    
    print("height:", p1.get(4))
    print("width:", p1.get(3))
    print("total frame:", p1.get(7))
    #print("FPS:", p1.get(5))

    #start_frame = 400
    #start_frame = 950
    #start_frame = 2560
    #start_frame = 100 #930 #700 #280 #100
    start_frame = 0

    end_frame = start_frame + 900.0
    p1.set(1, start_frame) # set current frame
    while p1.isOpened()==True:
        ret, frame_ori = p1.read()
        #p1.set(1, p1.get(1)+10.)
        if ret == True:
            frame_no = p1.get(1)
            print("frame:", frame_no, end=" ")
            if frame_no > end_frame:
                break
            start_time = time.time()
            frame = cv2.resize(frame_ori, (960, 540)) #resize
            #frame = frame_ori
            # detect faces in a frame, and preprocess
            prepro_faces, raw_results = preprocess_image(frame)
            
            #print(prepro_faces, raw_results, len(prepro_faces), len(raw_results))
            if len(prepro_faces) == 0:
                continue
                
            time_03 = time.time()   
            #
            # Inference, 1st output is probability of age, 2nd probability of gender
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

            # plot results of inference
            for i in range(len(prepro_faces)):
                pred_age = np.array(pred[0][i]).argmax(axis=-1)
                pred_gender = np.array(pred[1][i]).argmax(axis=-1)

                #x1, y1, width, height = raw_results[i]['box']  #for MTCNN
                x1, y1, width, height = raw_results[i]
                cv2.rectangle(frame, (x1,y1), (x1+width, y1+height), (0,0,255), 2)
                #text = "{},{}  {}x{}".format(cls2age[pred_age], cls2gender[pred_gender], width, height)
                text = "{},{}".format(cls2age[pred_age], cls2gender[pred_gender])
                cv2.putText(frame, text, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # plot ID
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
            if cv2.waitKey(33) != -1: # press any key to escape(if not press then cv2.waitKey() will return -1)
                break
        else:
            break

        #  DataFrame  df_age_gender write to DB 
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
        # end of while loop


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
