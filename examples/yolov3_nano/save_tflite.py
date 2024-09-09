import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.layers import UpSampling2D,Concatenate,LeakyReLU
from tensorflow.keras.models import Model
import argparse

from utils import get_random_data
from model import tiny_yolo_res_body
from tflite_add_post_processing import add_post_node
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('-weight', help='trained weight', default=r'yolo3_128_128_0.25_nano_final_weights.h5',type=str)

    parser.add_argument('-height', help='trained weight', default=128,type=int)
    parser.add_argument('-width', help='trained weight', default=128,type=int)
    parser.add_argument('-num_heads', help='trained weight', default=3,type=int)
    parser.add_argument('-alpha', help='trained weight', default=0.25,type=float)
    parser.add_argument('-anchor', help='test image', default=r'./160_128_anchors.txt',type=str)

    parser.add_argument('-tflite', help='test image', default=r'yolo3_128_128_0.25_nano_final.tflite',type=str)
    
    args, unknown = parser.parse_known_args()

    input_shape = (args.height,args.width,3)
    m = tiny_yolo_res_body(input_shape,3,1,args.alpha,weight_load=False)
   
    m.load_weights(args.weight)

    converter = tf.lite.TFLiteConverter.from_keras_model(m)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supportes_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    annotation_path = 'train.txt'

    with open(annotation_path) as f:
        annotation_lines = f.readlines()
    annotation_lines = annotation_lines[0:500]
    def representative_data_gen():
        '''data generator for fit_generator'''
        n = len(annotation_lines)
        i = 0
        input_shape = (args.height,args.width)
        
        image_data = []
        box_data = []
        for b in range(100):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image = (image).astype('float32')
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        #image_data = np.array(image_data).astype('int8')
        image_data = np.array(image_data)
        
        for input_value in image_data:
            input_value = input_value.reshape(1,args.height,args.width,3)
            yield [input_value]

    converter.representative_dataset = representative_data_gen
    tflite_model_quant = converter.convert() 

    with open(args.tflite,'wb') as f:
        f.write(tflite_model_quant)
        f.close()

    anchors = get_anchors(args.anchor)
    
    if(len(anchors.flatten()) // 6 ==3):
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    elif (len(anchors.flatten()) // 6 == 2):
        anchor_mask = [[3,4,5], [0,1,2]]
    anchors = anchors[anchor_mask]
    add_post_node(args.tflite,anchors,(args.height,args.width))

    print('save %s done'%args.tflite)
