import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import backend as K
from keras.layers import Input,Lambda
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard,ModelCheckpoint
import shutil
import threading

from utils import get_random_data,yolo_cfg
from utils import mbv1_pretrained_weights
from model import tiny_yolo_res_body,yolo_loss
from tflite_add_post_processing import add_post_node
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def create_res_tiny_model(input_shape, anchors, num_classes,
            weights_path='model_data/tiny_yolo_weights.h5',weight_load = False,alpha=1.0,
            iou_threshhold=0.5,divider=[32,16,8]):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    h, w = input_shape
    
    image_input = Input(shape=(h, w, 3))
    if(weight_load):
        print('Load weights {}.'.format(weights_path))
    num_anchors = len(anchors)
    
    
    y_true = [Input(shape=(h//{0:divider[0], 1:divider[1],2:divider[2]}[l], w//{0:divider[0], 1:divider[1], 2:divider[2]}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]
    model_body = tiny_yolo_res_body((h, w, 3), num_anchors//3, num_classes,alpha=alpha,
    weights_path = weights_path,weight_load=weight_load)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': iou_threshhold})(
        [*model_body.output, *y_true])

    model = Model([model_body.input, *y_true], model_loss)
        
    return model,model_body

def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes,divider):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting
    if (num_layers >1):
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5],[0,1,2]]
    elif num_layers == 1: 
        anchor_mask = [[0,1,2]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    m = true_boxes.shape[0]
    if (num_layers == 3):
        grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
        y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
            dtype='float32') for l in range(num_layers)]
    elif num_layers == 2:
        grid_shapes = [input_shape//{0:divider[0], 1:divider[1]}[l] for l in range(num_layers)]
        y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
            dtype='float32') for l in range(num_layers)]
    elif num_layers == 1:
        grid_shapes = [input_shape//divider[0]]
        y_true = [np.zeros((m,grid_shapes[0][0],grid_shapes[0][1],len(anchor_mask[0]),5+num_classes),
        dtype='float32')]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true


class multi_thread(threading.Thread):
    def __init__(self, func, args=()):
        super(multi_thread, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.result = self.func(*self.args)
    def update_args(self,func,args=()):
        self.args = args
        self.func = func
    def get_result(self):
        threading.Thread.join(self)
        try :
            return self.result
        except Exception as e:
            print("error: "+str(e))
            exit()

def multi_get_random_data(annotation_lines, input_shape, random=True):
    n = len(annotation_lines)
    image_list = []
    box_list = []
    for i in range(n):
        image, box = get_random_data(annotation_lines[i], input_shape, random=True)
        image_list.append(image)
        box_list.append(box)

    return image_list, box_list

def multi_thread_data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes,divider,thread_num=8):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    
    while True:
        
        image_data = []
        box_data = []
        lines = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            lines.append(annotation_lines[i])    
            i = (i+1) % n
        step = batch_size // thread_num
        line_list = [lines[j:j+step] for j in range(0,len(lines),step)]
        
        threads = []
        for l in range(thread_num):
            threads.append(multi_thread(multi_get_random_data,(line_list[l],input_shape,True))) 
            
        for l in range(thread_num):
            threads[l].start()
        
        for l in range(thread_num):
            if l == 0:
                image_data, box_data = threads[l].get_result()
            else:
                image, box = threads[l].get_result()
                image_data.extend(image)
                box_data.extend(box)
        
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes,divider)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes,divider):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    
    while True:
        
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes,divider)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes,divider,thread_num=8):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return multi_thread_data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes,divider,thread_num)


def find_weights(net_w,net_h,net_alpha):
    net_size = min(net_w,net_h)
    for weight in mbv1_pretrained_weights:
        if [net_size,net_size,net_alpha] == weight:
            return True, './weights/mbv1_%s_%d_%d.h5'%(str(net_alpha),net_size,net_size)
    
    return False,None



class LRCosWarmUpRestart(keras.callbacks.Callback):

    def __init__(self, min_learn_rate, warmEpoch, max_learn_rate,cosHalfCycleEpoch,restart_times):
        
        super(LRCosWarmUpRestart, self).__init__()

        self.lr_list = []
        self.min_learn_rate = min_learn_rate
        self.warmEpoch = warmEpoch
        self.max_learn_rate = max_learn_rate
        self.cosHalfCycleEpoch = cosHalfCycleEpoch
        self.restart_times = restart_times


    def LRCalc(self,epoc, lr):
        warmEpoch = self.warmEpoch
        maxLR = self.max_learn_rate
        if epoc < warmEpoch:
            lr = (maxLR - self.min_learn_rate) / (warmEpoch + 1) * (epoc + 1) + self.min_learn_rate
        else:
            epoc -= warmEpoch
            ndx = epoc % self.cosHalfCycleEpoch
            k = self.cosHalfCycleEpoch
            amp = (1 + (epoc // self.cosHalfCycleEpoch) * 1.5)
            lr = maxLR * (1 + np.math.cos(np.pi / k * ndx)) / 2 / amp + self.min_learn_rate
        return lr

    def on_epoch_begin(self, epoch, logs={}):
        self.lr = K.get_value(self.model.optimizer.lr)
        self.lr = self.LRCalc(epoch,self.lr)
        K.set_value(self.model.optimizer.lr, self.lr)
        print("\nSet LR to %f"%self.lr)
        self.lr_list.append(self.lr)


class LossHistory(keras.callbacks.Callback):
    """
    
    """
    def __init__(self):
        super().__init__()

    def on_train_begin(self, logs=None):
        self.train_loss = {'batch':[],'epoch':[]}
        self.val_loss = {'batch':[],'epoch':[]}
        self.min_val_loss = 500.0
        self.min_train_loss = 500.0
        
    def on_train_end(self, logs=None):
        print('train end')
        

    def on_epoch_end(self, epoch, logs={}):
        self.train_loss['epoch'].append(logs.get('loss'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        if (logs.get('val_loss') < self.min_val_loss):
            self.min_val_loss = logs.get('val_loss')
            self.min_train_loss = logs.get('loss')
            self.best_epoch = epoch
            print("\n%d Min loss:%f, val loss:%f"%(self.best_epoch,self.min_train_loss,self.min_val_loss))
            self.model.save_weights("./best_loss_weight.h5")

def model_train(cfg):
    
    log_dir = 'logs/000/'
    anchors_path = cfg.cluster_anchor
    class_names = cfg.class_names
    num_classes = cfg.num_classes
    anchors = get_anchors(anchors_path)
    divider = cfg.divider
    input_shape = (cfg.height,cfg.width) # multiple of 32, hw
    alpha = cfg.alpha
    
    h,w = input_shape

    log_dir = 'logs/%d_%d_%s/'%(cfg.width,cfg.height,alpha)
    if cfg.load_weight == True:
        support_weight,weight =  find_weights(cfg.width,cfg.height,alpha) 
    else:
        support_weight = False
        weight = ''

    model,infer_model = create_res_tiny_model(input_shape, anchors, num_classes,iou_threshhold=cfg.iou_threshold,
    divider=divider,alpha = alpha, weights_path=weight,weight_load=support_weight)
    model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})
    model.summary()
    print('Build Model')

    try:
        shutil.rmtree(log_dir)
    except:
        print()
    model_name = 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + model_name,
        monitor='val_loss', save_weights_only=True, save_best_only=True, verbose=1)

    val_split = 0.1
    with open('train.txt','r') as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    '''
    '''
    batch_size = cfg.batch_size 
    warmEpoch = cfg.total_epochs // 10
    restart_times = 3
    cosHalfCycleEpoch = (cfg.total_epochs-cfg.total_epochs // 10) // restart_times
    min_lr_rate = 1e-8
    max_lr_rate = 0.0025
    lr_warmup_restart = LRCosWarmUpRestart(min_lr_rate,warmEpoch,max_lr_rate,cosHalfCycleEpoch,restart_times)
    loss_his = LossHistory()
    data_gen_thread_count = cfg.thread_count
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes,divider,data_gen_thread_count),
        steps_per_epoch=max(1, num_train//batch_size),
        validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes,divider,data_gen_thread_count),
        validation_steps=max(1, num_val//batch_size),
        epochs=cfg.total_epochs,
        initial_epoch=0,
        callbacks=[logging, checkpoint, lr_warmup_restart,loss_his])#,max_queue_size=12,workers=8,use_multiprocessing=True)
    
    print('Training Complete best epoch:%d loss:%f-val:%f'%(loss_his.best_epoch,loss_his.min_train_loss,loss_his.min_val_loss))
    model.load_weights('best_loss_weight.h5')
    model.save_weights('yolo3_%d_%d_%s_nano_final_weights.h5'%(cfg.width,cfg.height,alpha))

    infer_model.load_weights('yolo3_%d_%d_%s_nano_final_weights.h5'%(cfg.width,cfg.height,alpha))
    converter = tf.lite.TFLiteConverter.from_keras_model(infer_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supportes_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    def representative_data_gen():
        '''data generator for fit_generator'''
        n = len(lines)
        i = 0
        input_shape = (cfg.height,cfg.width)
        image_data = []
        box_data = []
        for b in range(100):
            if i==0:
                np.random.shuffle(lines)
            image, box = get_random_data(lines[i], input_shape, random=True)
            image = (image).astype('float32')
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n

        image_data = np.array(image_data)
        
        for input_value in image_data:
            input_value = input_value.reshape(1,cfg.height,cfg.width,3)
            yield [input_value]

    converter.representative_dataset = representative_data_gen
    tflite_model_quant = converter.convert() 
    m_path = 'yolo3_%d_%d_%s_nano_final.tflite'%(cfg.width,cfg.height,alpha)
    with open(m_path,'wb') as f:
        f.write(tflite_model_quant)
        f.close()
    
    if cfg.num_heads == 3:
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    elif cfg.num_heads == 2:
        anchor_mask = [[3,4,5], [0,1,2]]
    else:
        anchor_mask = [[0,1,2]]
    anchors = anchors[anchor_mask]
    add_post_node(m_path,anchors,(cfg.height,cfg.width))
    
    np.save("lr_%d_%d_%s.npy"%(cfg.width,cfg.height,alpha),np.array(lr_warmup_restart.lr_list))
    np.save("val_loss_%d_%d_%s.npy"%(cfg.width,cfg.height,alpha),np.array(loss_his.val_loss['epoch']))
    np.save("train_loss_%d_%d_%s.npy"%(cfg.width,cfg.height,alpha),np.array(loss_his.train_loss['epoch']))
    


if __name__ == '__main__':

    cfg = yolo_cfg()
    model_train(cfg)


    
