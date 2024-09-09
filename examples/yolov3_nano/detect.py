import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
import numpy as np 
from utils import yolo_cfg
import argparse
from PIL import Image,ImageDraw
from evaluate import get_yolo_boxes

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('-model', help='trained tflite', default=r'face_yolo3_160_128_0.25_nano_final.tflite',type=str)
    parser.add_argument('-image', help='test image', default=r'./person.jpeg',type=str)
    parser.add_argument('-anchor', help='test image', default=r'./face_160_128_anchors.txt',type=str)
    args, unknown = parser.parse_known_args()

    labels = ['person']
    cfg = yolo_cfg()
    
    interpreter = tf.lite.Interpreter(model_path=str(args.model))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    shape = input_details['shape']

    cfg.width = shape[2]
    cfg.height = shape[1]
    anchors = get_anchors(args.anchor)
    anchors_num = len(anchors) / 3


    image = Image.open(args.image)
    origin_image = image
    image_w = image.size[0]
    image_h = image.size[1]
    
    scale = min(cfg.width/image_w, cfg.height/image_h)
    nw = int(image_w*scale)
    nh = int(image_h*scale)
    dx = (cfg.width-nw)//2
    dy = (cfg.height-nh)//2
    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', (cfg.width,cfg.height), (128,128,128))
    new_image.paste(image, (dx, dy))
    
    image_data = np.array(new_image)/255.
    image_data = image_data.reshape(1,cfg.width,cfg.height,3)


    pred_boxes,scores,classes = get_yolo_boxes(args.model,anchors,image_data,(image_h,image_w),0.1,0.35)

    im = origin_image
    draw = ImageDraw.Draw(im)
    for i in range(len(pred_boxes)): 
        box = pred_boxes[i]
        score = "%.3f"%scores[i]
        
        draw.rectangle([box[0],box[1],box[2],box[3]],outline=tuple(np.random.randint(0, 255, size=[3])), width=2)
        draw.text([box[0],box[1]], score)
    im.save('tflite_detected_img.jpg')
    im.show()
    print('after predict')



