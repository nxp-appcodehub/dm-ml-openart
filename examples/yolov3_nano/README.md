# YOLOV3 Nano

A light weight yolov3 implemation for MCU platforms.

## Introduction

MCU platforms has limited ram and flash, which could not running original yolo models even tiny version. In this project we use mbv1 net instead of darknets and down size the channel number of the header. we keep use 3 heads to detect 3 scales of objects. Flash uasge reduced from 9.6M to 240KB, RAM usage from 6M to 423KB.

Support training on Pascal or COCO dataset, and custorm VOC type dataset as well.

## Requirements

tensorflow 2.10

numpy

PILLOW

matplotlib

configparser

json

opencv-python

## Model performance

Person detect model performance :

| input size / alpha | Flash(KB) | RAM(KB) | PASCAL mAP |
| ------------------ | --------- | ------- | ---------- |
| 128x128_0.25       | 240       | 158     | 0.37       |
| 128x128_0.5        | 296       | 293     | 0.43       |
| 128x160_0.25       | 240       | 230     | 0.33       |
| 128x160_0.5        | 296       | 438     | 0.4        |
| 160x160_0.25       | 240       | 230     | 0.39       |
| 160x160_0.5        | 296       | 438     | 0.47       |
| 160x192_0.25       | 240       | 319     | 0.39       |
| 160x192_0.5        | 296       | 615     | 0.47       |
| 192x192_0.25       | 240       | 319     | 0.41       |
| 192x192_0.5        | 296       | 615     | 0.48       |
| 224x224_0.25       | 245       | 423     | 0.37       |
| 224x224_0.5        | 302       | 824     | 0.52       |

## Train

* edit config.cfg
* python voc_convertor.py
* python kmeans_plus.py
* python train.py

## Evaluate

* python evaluate.py -model xxx.tflite -anchor yolo3_anchors.txt

## Demo shows

### Face detect demo

<video width="320" height="240" controls>
  <source src="docs\face_det.mp4" type="video/mp4">
</video>

### Person detect demo

<video width="320" height="240" controls>
  <source src="docs\person_det.mp4" type="video/mp4">
</video>
