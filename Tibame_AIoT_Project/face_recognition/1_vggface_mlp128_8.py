# -*- coding: utf-8 -*-
"""1_vggface_mlp128-8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MS3KCIfVZ_msuBAg6QYMstOkThsApdMl
"""

# age 分成8個 classes
# mlp 每個全連接層的unit個數: 128 - 8
# trainning: 
#   改用generator產生資料給fit_generator
#   class_weight
#   random_state
#   callback: Early Stop, model.save


#用少量資料
FULL_DATA = 0
per_cls = 600
#用全部資料
#FULL_DATA = 1
IMG_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 2

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# to measure execution time
!pip install ipython-autotime
# %load_ext autotime

! nvidia-smi

!pip install mtcnn

# Commented out IPython magic to ensure Python compatibility.
import scipy.io
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

#import tensorflow as tf
#from tensorflow import keras
import keras
from keras.preprocessing.image import load_img
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Activation
from keras.layers import Conv2D, AveragePooling2D, BatchNormalization
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras import metrics

import matplotlib.pyplot as plt
# %matplotlib inline

from keras.models import load_model
import cv2
from glob import glob
import os
from mtcnn import MTCNN
import numpy as np

df = pd.read_csv('drive/My Drive/Tibame_AIoT_Project/Datasets/資料集_IMDB-Wiki/age_gender_wiki.csv')
df_under10 = pd.read_csv('drive/My Drive/Tibame_AIoT_Project/Datasets/資料集_IMDB-Wiki/age_gender_imdb_under10.csv')
df_over70 = pd.read_csv('drive/My Drive/Tibame_AIoT_Project/Datasets/資料集_IMDB-Wiki/age_gender_imdb_over70.csv')

df = pd.concat([df, df_under10, df_over70])

#some guys seem to be greater than 100. some of these are paintings. remove these old guys
df = df[df['age'] <= 100]
 
#some guys seem to be unborn in the data set
df = df[df['age'] > 0]

# 每10歲分一類,70歲以上歸為同一類,共8類
df['age_grp'] = pd.cut(df['age'], bins=[0,10,20,30,40,50,60,70,110], right=False)
le = LabelEncoder()
le.fit(df['age_grp'].astype('str'))
df['age_cls'] = le.transform(df['age_grp'].astype('str'))
df

df['age_cls'].value_counts().sort_index()

histogram_age = df['age_cls'].hist(bins=df['age_cls'].nunique())

#先用少量資料比較不同模型:
#每個類別各取部分資料,用train_test_split來切train and test
df_0 = df[df['age_cls'] == 0]
df_1 = df[df['age_cls'] == 1]
df_2 = df[df['age_cls'] == 2]
df_3 = df[df['age_cls'] == 3]
df_4 = df[df['age_cls'] == 4]
df_5 = df[df['age_cls'] == 5]
df_6 = df[df['age_cls'] == 6]
df_7 = df[df['age_cls'] == 7]
# train and val data
if FULL_DATA == 1:
    train_df = pd.concat([df_0[:-100], df_1[:-100], df_2[:-100], df_3[:-100], 
        df_4[:-100], df_5[:-100], df_6[:-100], df_7[:-100] ])    
else:    
    #先用少量資料比較不同模型
    train_df = pd.concat([df_0[:per_cls], df_1[:per_cls], df_2[:per_cls], df_3[:per_cls], 
        df_4[:per_cls], df_5[:per_cls], df_6[:per_cls], df_7[:per_cls] ])
# predict data: 每個類別保留最後100筆資料作為predict用
predict_df = pd.concat([df_0[-100:], df_1[-100:], df_2[-100:], df_3[-100:], 
        df_4[-100:], df_5[-100:], df_6[-100:], df_7[-100:] ])
x_pre, y_pre = np.array(predict_df['full_path']), np.array(predict_df['age_cls'])
print("train:", len(train_df), "predict:", len(predict_df))

# 處理答案 把它轉成one-hot (後面再做)
# y_train_category = to_categorical(df['age_cls'], num_classes=8)

# 切分訓練data
x_train, x_test, y_train, y_test = train_test_split(np.array(train_df['full_path']), np.array(train_df['age_cls']), test_size=0.2, random_state=0)

print(x_train[0], x_test[0], y_train[0], y_test[0])
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

folder_path = 'drive/My Drive/Tibame_AIoT_Project/Datasets/資料集_IMDB-Wiki'
detector = MTCNN()
#feature_extractor = load_model(os.path.join(folder_path, 'facenet.h5'))
#feature_extractor = load_model(os.path.join(folder_path, 'facenet_keras.h5'))

# VGGFace: https://github.com/rcmalli/keras-vggface
!pip install keras_vggface
!pip install keras_applications

from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
feature_extractor = VGGFace(model='resnet50', include_top=False, 
            input_shape=(224, 224, 3), pooling='avg')

feature_extractor.summary()

# 固定pre-train model的參數
for lyr in feature_extractor.layers:
    lyr.trainable = False

# BN
x = BatchNormalization()(feature_extractor.output)    
    
# MLP    
# x = Flatten()(x)

#x = Dense(units=2048, activation='relu')(x)
#x = Dense(units=512, activation='relu')(x)
x = Dense(units=128, activation='relu')(x)
x = Dense(units=8, activation='softmax')(x)
age_model = Model(inputs=feature_extractor.input, outputs=x)   
age_model.summary()

age_model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

model_folder_path = 'drive/My Drive/Tibame_AIoT_Project/face'
age_model.load_weights(os.path.join(model_folder_path,'1_vggface_weight_mlp128-8_cls600.h5'))

# 資料預處理 for facenet?
# Standardization
def preprocess(imgs): 
    for i in range(imgs.shape[0]):
        # standardization
        img = imgs[i]
        mean, std = img.mean(), img.std()
        img = (img - mean) / std
        imgs[i] = img
    return imgs
# Normalization
def normalize(img):
    return img / 255.

# -1 <= x <= 1
def preprocess_1(imgs):
    x = np.array(imgs, dtype = float)
    x /= 127.5
    x -= 1.
    return x

# detect face
def detect_faces(img):
    face_imgs = []
    results = detector.detect_faces(img)
    # extract the bounding box from the first face
    # print('# of faces: ', len(results))
    for i in range(len(results)):
        x1, y1, width, height = results[i]['box']
        x2, y2 = x1 + width, y1 + height
        patch = img[y1:y2, x1:x2] # crop face
        face_imgs.append(patch)
    return face_imgs

img_folder_path = 'drive/My Drive/Tibame_AIoT_Project/Datasets/資料集_IMDB-Wiki'

def data_generator(data_paths, y, batch_size=BATCH_SIZE):
    '''data generator for fit_generator'''
    n = len(data_paths)
    i = 0
    data_paths = data_paths
    
    while True:
        x_ori, x_norm, y_ori = [], [], []
        i_batch = i
        for b in range(batch_size):
            path = data_paths[i]
            #print("idx:", i, "cls:", y[i], path)
        
            # 讀取圖片,切下臉的部分,並使用借來的模型的預處理方式來作預處理           
            img = cv2.imread(os.path.join(img_folder_path,path))[:,:,::-1]
            
            # plt.imshow(img)
            # plt.show()
            faces = detect_faces(img)
            if len(faces) == 0 or faces[0].shape[0] == 0:
                print('No face')
                i = (i+1) % n
                continue   
            img_crop = cv2.resize(faces[0], (IMG_SIZE, IMG_SIZE))
            # plt.imshow(faces[0])
            # plt.show()

            # 使用借來的模型的預處理方式來作預處理
            img_pre = preprocess_input(np.array(img_crop,dtype=float))

            # 把原圖留下來
            x_ori.append(img)
            x_norm.append(img_pre)
            y_ori.append(y[i])
            
            i = (i+1) % n

        # print("len(image_data)",len(x_ori))
        # plt.figure(figsize=(10, 40))
        # for j,m in enumerate(x_ori):
        #     plt.subplot(1, BATCH_SIZE, (j%BATCH_SIZE)+1)
        #     plt.title("idx:{} y:{}".format(i_batch+j, y[i_batch+j]))
        #     plt.axis("off")
        #     plt.imshow(m)
        # plt.show()    
        y_category = to_categorical(y_ori, num_classes=8)    
        yield np.array(x_norm), np.array(y_category)

# 用generator產生資料
generator_train = data_generator(x_train, y_train, batch_size=BATCH_SIZE)
generator_test = data_generator(x_test, y_test, batch_size=BATCH_SIZE)
type(generator_train)

if FULL_DATA == 1:
    weights = {0:12., 1:5., 2:1., 3:2., 4:3., 5:4., 6:6., 7:3.}
else:    
    # for temp
    weights = {0:1., 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1.}

# fit_generator
checkpoint = ModelCheckpoint("1_vggface_mlp128-8_epoch.h5", save_best_only=False)   #Defaults: save_freq='epoch'
earlystop = EarlyStopping(patience=5, restore_best_weights=True)
logs = age_model.fit(generator_train,
                epochs=EPOCHS,
                steps_per_epoch=len(x_train)//BATCH_SIZE,
                validation_data=generator_test,
                validation_steps=len(x_test)//BATCH_SIZE,
                class_weight=weights,
                #validation_split=0.1,
                callbacks=[checkpoint, earlystop] 
                )

model_folder_path = 'drive/My Drive/Tibame_AIoT_Project/face'
age_model.save_weights(os.path.join(model_folder_path,'1_vggface_weight_mlp128-8_cls600.h5'))

cur_train_idx = 0
cur_test_idx = 0
def get_data(x, y, batch=20, IMG_SIZE=160, test=1):
    # 要注意 numpy 中的 randint 的上限是不包含的 和一般的randint不同
    # numpy array 的索引可以是個 list, 即可同時取出不只一個元素
    global cur_train_idx, cur_test_idx
    print("cur train/test idx:", cur_train_idx, cur_test_idx)    
    if test == 0:
        #idx = np.random.randint(0, len(x), batch)
        idx = list(range(cur_train_idx, cur_train_idx+batch, 1))
        cur_train_idx = (cur_train_idx + batch) % len(x)
    else:
        idx = np.random.randint(0, len(x), batch)
        #idx = list(range(cur_test_idx, cur_test_idx+batch, 1))
        cur_test_idx += batch

    #print("idx:", idx, x[idx], y[idx])
    x_idx = x[idx]
    y_idx = y[idx]
    x_ori, x_norm, y_ori = [], [], y_idx
    for i,p in enumerate(x_idx):
        print(p)
        # print(y_idx[i].argmax(axis=-1))
        # 讀取圖片並使用借來的模型的預處理方式來作預處理
        # img = np.array(load_img(os.path.join(img_folder_path,p[0]), target_size=(224, 224)))
        # 讀取圖片並切下臉的部分
        img = np.array(cv2.imread(os.path.join(img_folder_path,p))[:,:,::-1])
        # plt.imshow(img)
        # plt.show()
        faces = detect_faces(img)
        if len(faces) == 0 or faces[0].shape[0] == 0:
            print('No face')
            continue   
        img = cv2.resize(faces[0], (IMG_SIZE, IMG_SIZE))
        # plt.imshow(faces[0])
        # plt.show()

        # 使用借來的模型的預處理方式來作預處理
        img_pre = preprocess_input(np.array(img,dtype=float))
        #img_pre = preprocess_1(img)
        #img_pre = normalize(img)
        
        # 把原圖留下來
        x_ori.append(img)
        x_norm.append(img_pre)
    return np.array(x_ori), np.array(x_norm), np.array(y_ori)

# 取出要用來預測的資料
x_ori_batch, x_batch, y_batch = get_data(x_pre, y_pre, batch=20, IMG_SIZE=224) 
print(y_batch)

# predict
pre = age_model.predict(x_batch).argmax(axis=-1)
pre

from sklearn.metrics import confusion_matrix
pd.DataFrame(confusion_matrix(y_batch, pre),
            index=["{}(真實)".format(i) for i in range(8)],
            columns=["{}(預測)".format(i) for i in range(8)] 
            )



def euclidean_distance(x, y):
    sum_square = np.sum(np.square(x - y), keepdims=True)
    return np.sqrt(sum_square)

def predict_age(img):
    img_size = 100
    img = normalize(img)
    img = cv2.resize(img, (img_size, img_size))
    model_input = np.zeros((1, img_size, img_size, 3))
    model_input[0] = img
    ages = age_model.predict(model_input)
    print('age: ', ages.argmax(axis=-1))
    return 

# def predict_gender(img):
#     img_size = 100
#     img = normalize(img)
#     img = cv2.resize(img, (img_size, img_size))
#     model_input = np.zeros((1, img_size, img_size, 3))
#     model_input[0] = img
#     genders = model_gender.predict(model_input)
#     gender = genders[0]
#     if gender > 0.5:
#         print('Male')
#     else:
#         print('Female')
#     return

def face_id(filename, IMG_SIZE=160):
    raw_img = cv2.imread(os.path.join(folder_path, filename))[:,:,::-1]
    faces = detect_faces(raw_img)
    if len(faces) == 0:
        print('No face')
        return
    else:
        # get face embeddings
        face = faces[0]
        # More predictions
        predict_age(face)
        # predict_emotion(face)
        # predict_gender(face)
        # # ID
        # face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        # model_input = np.zeros((1, IMG_SIZE, IMG_SIZE, 3))
        # model_input[0] = face
        # model_input = preprocess(model_input)
        # query_embeddings = feature_extractor.predict(model_input)
        # query_embedding = query_embeddings[0]
        
        # # compute distance
        # distances = np.zeros((len(embeddings)))
        # for i, embed in enumerate(embeddings):
        #     distance = euclidean_distance(embed, query_embedding)
        #     distances[i] = distance

        # # find min distance    
        # idx_min = np.argmin(distances)
        # distance, name = distances[idx_min], names[idx_min]
        # print('name: ', name, ' distance: ',distance)

folder_path = '/content/drive/My Drive/week10/face_detection'
path = 'face3.jpg'
face_id(path)
plt.imshow(cv2.imread(os.path.join(folder_path, path))[:,:,::-1])

