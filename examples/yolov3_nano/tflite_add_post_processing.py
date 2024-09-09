from ctypes import *
import platform
import sys
from utils import yolo_cfg

def add_post_node(model,anchors,shape):

    class post_param(Structure):
        _fields_ = [
            ("width",c_int),
            ("height",c_int),
            ("num_classes",c_int),
            ("num_heads",c_int),
            ("max_detections",c_int),
            ("anchors",c_int*18),
            ("nms_iou_threshold",c_float),
            ("nms_score_threshold",c_float),
            ("model_file",c_wchar_p),
        ]

    if sys.platform == 'linux':
        lib = './add_post_processing_64.so'
    elif sys.platform == 'win32':
        arch = platform.architecture()
        if arch[0] == '64bit':
            lib = './add_post_processing_64.dll'
        else:
            lib = './add_post_processing_32.dll'
            print("32bit system not support")
            exit()

    lib = CDLL(lib)
    lib.add_node.argtypes = [c_void_p]

    param = post_param()
    cfg = yolo_cfg()

    param.width = shape[1]
    param.height = shape[0]
    param.num_classes = cfg.num_classes
    param.num_heads = cfg.num_heads
    param.max_detections = cfg.num_heads
    
    anchors = anchors.flatten().astype('int32')
    for i in range(len(anchors)):
        param.anchors[i] = anchors[i]
    param.nms_iou_threshold = cfg.nms_iou_threshold
    param.nms_score_threshold = cfg.nms_score_threshold
    model_files = create_string_buffer(256)
    model_files= bytes(model,'utf-8')

    ret = lib.add_node(pointer(param),model_files)

import argparse
import tensorflow as tf
import numpy as np
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('-model', help='trained tflite', default=r'yolo3_160_128_0.25_nano_final.tflite',type=str)
    parser.add_argument('-anchors', help='trained anchors', default=r'160_128_anchors.txt',type=str)
    args, unknown = parser.parse_known_args()

    anchors = get_anchors(args.anchors)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    anchors = anchors[anchor_mask]

    interpreter = tf.lite.Interpreter(model_path=str(args.model))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    shape = input_details['shape']
    w = shape[2]
    h = shape[1]
    
    add_post_node(args.model,anchors,(h,w))