import json
from collections import defaultdict
from utils import yolo_cfg
from PIL import Image
min_model_input_w = 128
min_model_input_h = 128

def box_filter(box,img_shape):
    (x_min,y_min,x_max,y_max) = box
    (img_w,img_h) = img_shape
    w = x_max - x_min
    h = y_max - y_min
    erea_ratio = (w*h) / (img_w*img_h)
    if  erea_ratio >= 0.01 and erea_ratio < 0.5:
        return True
    else:
        return False

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def convert_annotation(annotation,coco_path, cls_names, out_file):
    name_box_id = defaultdict(list)
    id_name = dict()

    miss_count = 0
    valid_count = 0
    f = open(
        annotation,
        encoding='utf-8')
    data = json.load(f)

    annotations = data['annotations']
    for ant in annotations:
        id = ant['image_id']
        name = '%s/%012d.jpg' % (coco_path,id)
        cat = ant['category_id']

        if cat >= 1 and cat <= 11:
            cat = cat - 1
        elif cat >= 13 and cat <= 25:
            cat = cat - 2
        elif cat >= 27 and cat <= 28:
            cat = cat - 3
        elif cat >= 31 and cat <= 44:
            cat = cat - 5
        elif cat >= 46 and cat <= 65:
            cat = cat - 6
        elif cat == 67:
            cat = cat - 7
        elif cat == 70:
            cat = cat - 9
        elif cat >= 72 and cat <= 82:
            cat = cat - 10
        elif cat >= 84 and cat <= 90:
            cat = cat - 11

        name_box_id[name].append([ant['bbox'], cat])

    f = open(out_file, 'w')
    for key in name_box_id.keys():
        box_infos = name_box_id[key]
        obj_count = 0
        for info in box_infos:
            label = class_names[int(info[1])]
            if label in cls_names:
                obj_count += 1
                break
        
        if obj_count == 0:
            continue

        buffer = key
        has_valid_box = False
        for info in box_infos:
            label = class_names[int(info[1])]
            if label not in cls_names:
                continue
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])
            img = Image.open(key)
            img_w,img_h = img.size
            if(box_filter((x_min,y_min,x_max,y_max),(img_w,img_h)) == True):
                has_valid_box = True
                buffer = buffer + "$%d,%d,%d,%d,%d" % (
                    x_min, y_min, x_max, y_max, int(info[1]))
        if (has_valid_box):     
            f.write(buffer)
            f.write('\n')
            valid_count = valid_count +1
        else:
            miss_count = miss_count + 1
    f.close()
    print("valid files:%d, invalid files:%d"%(valid_count,miss_count))

if __name__ == '__main__':
    cfg = yolo_cfg()
    cls_classes = cfg.class_names
    coco_path = cfg.voc_folder
    convert_annotation('%s/annotations/instances_train2017.json'%coco_path,'%s/train2017'%coco_path,cls_classes,'train.txt')
    convert_annotation('%s/annotations/instances_val2017.json'%coco_path,'%s/val2017'%coco_path,cls_classes,'test.txt')
    