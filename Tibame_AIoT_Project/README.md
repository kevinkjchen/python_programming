# AIoT專題實作
[第四組 AIoT人流分析系統](https://drive.google.com/file/d/1r07A2nqnBT2vzzC84VvdAU4mbS5jxwh9/view?usp=sharing)

## object_detection
訓練 yolo v4 模型來作 person 的 object detection 之前, 先用python程式檢查與整理標註檔

## face_classification
以人臉作age和gender的分類

## face_classification_demo
demo for age和gender的inference

## face_classification_demo/yolo4_replace_mtcnn
以 MTCNN 取得每一張臉的 bounding box, 轉成 yolo 的 label 並寫入txt檔, 用來訓練 yolo v4 模型來作 face 的 object detection