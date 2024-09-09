import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
import numpy as np 
import os 
from utils import yolo_cfg
import argparse
from PIL import Image,ImageDraw

from utils import get_random_data,yolo_cfg
from calc_mAP import get_map
def _sigmoid(x):
    return 1. / (1. + np.exp((-x)))

def decode_output(netout,input_shape,image_shape,anchors,conf_thres):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    net_h,net_w = input_shape
    image_h,image_w = image_shape
    boxes = []

    anchors = anchors.flatten()
    scores = netout[..., 4:5]
    classes = netout[..., 5:]
    scores = _sigmoid(scores)
    classes = _sigmoid(classes)

    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    #netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    
    classes = netout[..., 5:]

    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h*net_w)/image_w
    else:
        new_h = net_w
        new_w = (image_w*net_h)/image_h
    
    x_offset, x_scale = (net_w - new_w)/2./net_w, float(net_w)/new_w
    y_offset, y_scale = (net_h - new_h)/2./net_h, float(net_h)/new_h

    for i in range(grid_h*grid_w):
        row = int(i / grid_w)
        col = i % grid_w
        
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            #objectness = netout[..., :4]
            
            if(objectness <= conf_thres): continue
            
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]

            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height 
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height  
            
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            
            x = (x - x_offset) * x_scale
            y = (y - y_offset) * y_scale
            w *= x_scale
            h *= y_scale

            x2 = (x+w/2)*image_w
            y2 = (y+h/2)*image_h
            if x2 >= image_w: x2 = image_w-10
            if y2 >= image_h: y2 = image_h -10
            box = ((x-w/2)*image_w, (y-h/2)*image_h, x2, y2, objectness, classes)

            if (box[0] > 0) and (box[0] < box[2]) and (box[2] < image_w) and (box[1] > 0) and (box[1]< box[3]) and (box[3] < image_h):
                boxes.append(box)

    return boxes

def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua  
    
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap     

def apply_non_max_suppression(boxes, scores, iou_thresh=.449, top_k=200):
    """ non maximum suppression in numpy

    Arguments:
        boxes : array of boox coordinates of shape (num_samples, 4)
            where each columns corresponds to x_min, y_min, x_max, y_max
        scores : array of scores given for each box in 'boxes'
        iou_thresh : float intersection over union threshold for removing boxes
        top_k : int Number of maximum objects per class

    Returns:
        selected_indices : array of integers Selected indices of kept boxes
        num_selected_boxes : int Number of selected boxes
    """

    selected_indices = np.zeros(shape=len(scores))
    if boxes is None or len(boxes) == 0:
            return selected_indices
    x_min = boxes[:, 0]
    y_min = boxes[:, 1]
    x_max = boxes[:, 2]
    y_max = boxes[:, 3]
    areas = (x_max - x_min) * (y_max - y_min)
    remaining_sorted_box_indices = (np.argsort(scores)).astype('int32')
    remaining_sorted_box_indices = remaining_sorted_box_indices[-top_k:]

    num_selected_boxes = 0
    while len(remaining_sorted_box_indices) > 0:
            best_score_index = remaining_sorted_box_indices[-1]
            # print(len(remaining_sorted_box_indices), num_selected_boxes)
            # print(remaining_sorted_box_indices[::-1])
            selected_indices[num_selected_boxes] = best_score_index
            num_selected_boxes = num_selected_boxes + 1
            if len(remaining_sorted_box_indices) == 1:
                break

            remaining_sorted_box_indices = remaining_sorted_box_indices[:-1]

            best_x_min = x_min[best_score_index]
            best_y_min = y_min[best_score_index]
            best_x_max = x_max[best_score_index]
            best_y_max = y_max[best_score_index]

            remaining_x_min = x_min[remaining_sorted_box_indices]
            remaining_y_min = y_min[remaining_sorted_box_indices]
            remaining_x_max = x_max[remaining_sorted_box_indices]
            remaining_y_max = y_max[remaining_sorted_box_indices]

            inner_x_min = np.maximum(remaining_x_min, best_x_min)
            inner_y_min = np.maximum(remaining_y_min, best_y_min)
            inner_x_max = np.minimum(remaining_x_max, best_x_max)
            inner_y_max = np.minimum(remaining_y_max, best_y_max)

            # print(best_score_index, remaining_y_max[-1], inner_y_max[-1])

            inner_box_widths = inner_x_max - inner_x_min
            inner_box_heights = inner_y_max - inner_y_min

            inner_box_widths = np.maximum(inner_box_widths, 0.0)
            inner_box_heights = np.maximum(inner_box_heights, 0.0)

            intersections = inner_box_widths * inner_box_heights
            remaining_box_areas = areas[remaining_sorted_box_indices]
            best_area = areas[best_score_index]
            unions = remaining_box_areas + best_area - intersections

            intersec_over_union = intersections / unions
            intersec_over_union_mask = intersec_over_union <= iou_thresh
            remaining_sorted_box_indices = remaining_sorted_box_indices[
                                                intersec_over_union_mask]

    return selected_indices[0:num_selected_boxes].astype(int), 

def do_nmx_tf(box,iou_thres,num_classes):
    box = box.astype('float32')
    boxes = box[...,0:4]
    box_scores = box[...,4:]

    boxes_ = []
    scores_ = []
    classes_ = []
    max_boxes_tensor = K.constant(20, dtype='int32')
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.convert_to_tensor(boxes)
        class_box_scores = box_scores[...,0:1] * box_scores[...,1:]
        class_box_scores = K.concatenate(class_box_scores, axis=0)

        
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_thres)
        #nms_index = apply_non_max_suppression(boxes,class_box_scores,iou_thres)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_,scores_,classes_


def get_yolo_boxes(model,anchors,img_data,img_shape,conf_thres,iou_thres):
    interpreter = tf.lite.Interpreter(model_path=str(model))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    input_type = input_details['dtype']
    input_shape = input_details['shape']

    if input_type == np.uint8:
        test_data = (img_data * 255).astype('uint8')
    elif input_type == np.int8:
        test_data = (img_data*255-128).astype('int8')
    elif input_type == np.float32:
        test_data = img_data.astype('float32')

    

    test_data = test_data.reshape(input_shape)
    interpreter.set_tensor(input_details["index"], test_data)
    interpreter.invoke()
    
    pred_boxes = []
    out_len = len(output_details)
    if out_len > 1:
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if out_len==3 else [[3,4,5],[0,1,2]]
    else:
        anchors = anchors
        anchor_mask = [[0,1,2]]
    for i in range(out_len):
        output = interpreter.get_tensor(output_details[i]["index"])[0]
        net_shape = input_shape[1:3]
        if(output_details[i]['dtype'] == np.int8):
            zero_point = output_details[i]['quantization_parameters']['zero_points']
            scale = output_details[i]['quantization_parameters']['scales']
            output = ((output - zero_point)*scale).astype('float32')
        box = decode_output(output,net_shape,img_shape,anchors[anchor_mask[i]],conf_thres)
        pred_boxes += (box)
    if len(pred_boxes) > 0:
        pred_boxes,scores,classes = do_nmx_tf(np.array(pred_boxes),iou_thres,1)
        return np.array(pred_boxes),np.array(scores),np.array(classes)
    else:
        return [],[],[]
    

def del_file(filepath):
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

def evaluate(model,input_shape,annotation_lines,anchors,num_classes,score_threshold,iou_threshold,class_names,map_out_path):

    pred_scores = []
    n = len(annotation_lines)
    
    
    if not os.path.exists(map_out_path):
        os.mkdir(map_out_path)
    
    if not os.path.exists(os.path.join(map_out_path,"ground-truth/")):
        os.mkdir(os.path.join(map_out_path,"ground-truth/"))

    if not os.path.exists(os.path.join(map_out_path,"detection-results/")):
        os.mkdir(os.path.join(map_out_path,"detection-results/"))

    del_file(os.path.join(map_out_path,"ground-truth/"))
    del_file(os.path.join(map_out_path,"detection-results/"))
    for i in range(n):
        image, box = get_random_data(annotation_lines[i], input_shape, random=False)
        true_box = [b for b in box if b[2] > 0 and b[3] > 0]
        image_file = os.path.basename(annotation_lines[i].split('$')[0])
        image_id = image_file[0:image_file.rfind('.')]+'.txt'
        f = open(os.path.join(map_out_path,"ground-truth/"+image_id),'w')
        for b in true_box:
            f.write('%s %f %f %f %f\n'%(class_names[int(b[4])],b[0],b[1],b[2],b[3]))
            
        f.close()

        pred_boxes,pred_scores,pred_classes = get_yolo_boxes(model,anchors,image,input_shape,score_threshold,iou_threshold)
        f = open(os.path.join(map_out_path,"detection-results/"+image_id),'w')
        if(len(pred_boxes) > 0):
            for i in range(len(pred_boxes)):
                f.write("%s %f %f %f %f %f\n"%(class_names[int(pred_classes[i])],pred_scores[i],pred_boxes[i][0],pred_boxes[i][1],pred_boxes[i][2],pred_boxes[i][3]))
        else:
            f.write("%s 0.0 0.0 0.0 0.0 0.0\n"%(class_names[int(0)]))
        
        f.close()
        

    return 0    


        
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


pretrained_weights =[[128,128,0.25],[128,128,0.35],[128,128,0.5],[128,128,0.75],[128,128,1.0],
                    [160,160,0.25],[160,160,0.35],[160,160,0.5],[160,160,0.75],[160,160,1.0],
                    [192,192,0.25],[192,192,0.35],[192,192,0.5],[192,192,0.75],[192,192,1.0],
                    [224,224,0.25],[224,224,0.35],[224,224,0.5],[224,224,0.75],[224,224,1.0],
                   ]

def evaluate_all():
    mbv1_alpha= [
    0.25,0.5,
    #0.75,1.0
    ]
    size_supported = [
        [128,128],
        [128,160],
        [160,160],
        [160,192],
        [192,192],
    ]

    cfg = yolo_cfg()
    tflites = ['yolo3_iou_nano_final.tflite']
    
    anchors = [get_anchors(cfg.cluster_anchor),get_anchors(cfg.cluster_anchor),get_anchors(cfg.cluster_anchor)]
    print('evaluate begin')
    with open('test.txt','r') as f:
        lines = f.readlines()
        f.close()
    if not os.path.exists('./map_out'):
        os.mkdir('./map_out')

    for size in size_supported:
        h,w = size
        for i in range(len(mbv1_alpha)):
            alpha = mbv1_alpha[i]
            model = 'yolo3_%d_%d_%s_nano_final.tflite'%(w,h,alpha)
            
            if not os.path.exists(model):
                continue
            anchor = get_anchors('%d_%d_anchors.txt'%(w,h))

            interpreter = tf.lite.Interpreter(model_path=str(model))
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()[0]
            input_shape = input_details['shape']
            input_shape = input_shape[1:3]
            num_classes = cfg.num_classes
            
            score_threshold = 0.2

            map_out_path = './map_out/'+ model[0:model.rfind('.')] + '_out_map'
            evaluate(model,input_shape,lines,anchor,num_classes,score_threshold,0.35,cfg.class_names,map_out_path)
            map,score_average_list = get_map(MINOVERLAP=0.5,draw_plot=True,score_threhold = 0.3,path =map_out_path)
            print("%s mAP:%f average_socre: "%(model,map)+str(score_average_list))

    print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('-model', help='trained tflite', default=r'yolo3_160_128_0.25_nano_final.tflite',type=str)
    parser.add_argument('-anchor', help='test image', default=r'160_128_anchors.txt',type=str)
    parser.add_argument('-a','--all', help='test image', action='store_true')
    args, unknown = parser.parse_known_args()

    if args.all == True:
        evaluate_all()
    else:
        cfg = yolo_cfg()
        model = args.model
        anchor = get_anchors(args.anchor)

        with open('test.txt','r') as f:
            lines = f.readlines()
            f.close()
        if not os.path.exists('./map_out'):
            os.mkdir('./map_out')

        interpreter = tf.lite.Interpreter(model_path=str(model))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        input_shape = input_details['shape']
        input_shape = input_shape[1:3]
        num_classes = cfg.num_classes
        w = input_shape[1]
        
        score_threshold = 0.2

        map_out_path = './map_out/'+ model[0:model.rfind('.')] + '_out_map'
        evaluate(model,input_shape,lines,anchor,num_classes,score_threshold,0.35,cfg.class_names,map_out_path)
        map,score_average_list = get_map(MINOVERLAP=0.5,draw_plot=True,score_threhold = score_threshold,path =map_out_path)
        print("%s mAP:%f average_socre: "%(model,map)+str(score_average_list))