"""Miscellaneous utility functions."""

from functools import reduce

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import configparser


alpha_supported = [0.25,0.5,0.75,1.0]
size_supported = [128,160,192,224]
mbv1_pretrained_weights =[[128,128,0.25],[128,128,0.5],[128,128,0.75],[128,128,1.0],
                    [160,160,0.25],[160,160,0.5],[160,160,0.75],[160,160,1.0],
                    [192,192,0.25],[192,192,0.5],[192,192,0.75],[192,192,1.0],
                    [224,224,0.25],[224,224,0.5],[224,224,0.75],[224,224,1.0],
                   ]

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split('$')
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (0,0,0))
            new_image.paste(image, (dx, dy))
            image_data = (np.array(new_image))/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box
            for b in box:
                xmin,ymin,xmax,ymax,_ = b
                if(xmin > xmax) or (ymin > ymax) or xmax > w or ymax > h:
                    print(line,b)
        return image_data, box_data

    # resize image
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (0,0,0))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv((np.array(image))/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data

class yolo_cfg:
    def __init__(self):

        try:
            with open('./config.cfg','r') as fr:
                cfg = configparser.ConfigParser()
                cfg.readfp(fr)
                self.voc_folder = cfg['train']['voc_folder']
                self.test_folder = cfg['train']['test_folder']
                
                self.cluster_anchor = cfg['train']['cluster_anchor']
                self.total_epochs = int(cfg['train']['total_epochs'])
                self.batch_size = int(cfg['train']['batch_size'])
                self.thread_count = int(cfg['train']['thread_count'])
                
                self.cluster_number = int(cfg['model']['cluster_number'])
                self.width = int(cfg['model']['width'])
                self.height = int(cfg['model']['height'])
                self.divider = list(map(int,cfg['model']['divider'].split(",")))
                try:
                    self.load_weight = eval(cfg['model']['load_weight'])
                except:
                    print('"load_weight" config not boolean value')
                    self.load_weight = False
                self.alpha = float(cfg['model']['alpha'])
                if self.load_weight and  self.alpha not in alpha_supported:
                    print('unsupported alpha:%s in %s'%(cfg['model']['alpha'],alpha_supported))
                    exit()
 
                self.num_classes = int(cfg['model']['num_classes'])
                self.class_names = cfg['model']['class_names'].split(",")
                self.iou_threshold = float(cfg['model']['iou_threshold'])
                self.num_heads = 3
                self.nms_iou_threshold = float(cfg['inference']['nms_iou_threshold'])
                self.nms_score_threshold = float(cfg['inference']['nms_score_threshold'])
                self.max_detections = int(cfg['inference']['max_detections'])

            if not (self.num_heads *3 == self.cluster_number):
                print("Head number mismatch with anchor cluster count")
                exit()

        except Exception as e:
            print(str(e))
            exit()
