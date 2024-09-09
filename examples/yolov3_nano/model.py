import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model


def _make_divisible(v, divisor, min_value=None):
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v

def Relu6(x,max_value=6):
    return tf.keras.layers.ReLU(6.)(x)


def _conv_block(inputs, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.

    # Returns
        Output tensor.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = tf.keras.layers.Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
    return Relu6(x)


def mbv1_conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = tf.keras.layers.Conv2D(
        filters,
        kernel,
        padding='same',
        use_bias=False,
        strides=strides,
        name='conv1')(inputs)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return tf.keras.layers.ReLU(6., name='conv1_relu')(x)

def mbv1_depthwise_conv_block(inputs,
                          pointwise_conv_filters,
                          alpha,
                          depth_multiplier=1,
                          strides=(1, 1),
                          block_id=1):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = inputs
        x = tf.keras.layers.ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_%d' % block_id)(
            inputs)
    x = tf.keras.layers.DepthwiseConv2D((3, 3),
                                padding='same' if strides == (1, 1) else 'valid',
                                depth_multiplier=depth_multiplier,
                                strides=strides,
                                use_bias=False,
                                name='conv_dw_%d' % block_id)(
                                    x)
    x = tf.keras.layers.BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(
            x)
    x = tf.keras.layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = tf.keras.layers.Conv2D(
        pointwise_conv_filters, (1, 1),
        padding='same',
        use_bias=False,
        strides=(1, 1),
        name='conv_pw_%d' % block_id)(
            x)
    x = tf.keras.layers.BatchNormalization(
        axis=channel_axis, name='conv_pw_%d_bn' % block_id)(
            x)
    return tf.keras.layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

def mbv1_body(shape,alpha=1.0,depth_multiplier=1):
    input = tf.keras.layers.Input(shape=shape)
    x = mbv1_conv_block(input, 32, alpha, strides=(2, 2))
    x = mbv1_depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    x = mbv1_depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    x = mbv1_depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    x = mbv1_depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    
    first_block_filters = _make_divisible(256 * alpha, 8)
    model = Model(input, x) 
    return model,first_block_filters


def tiny_yolo_res_body(shape, num_anchors, num_classes,alpha=1.0, weights_path = './' ,weight_load=False):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    body,first_block_filters = mbv1_body(shape,alpha)
    
    if weight_load:
        body.load_weights(weights_path,by_name=True, skip_mismatch=True)
    
    for layer in body.layers:
        layer.trainable = False
    
    x1 = body.output    
    x2 = mbv1_depthwise_conv_block(x1,16,1,1,block_id=255)#//16
    x2 = mbv1_depthwise_conv_block(x2, 32,1,1, block_id=254)#//16
    x2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'valid')(x2)
    x3 = mbv1_depthwise_conv_block(x2, 16, 1,1,block_id=253)#//32
    x3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'valid')(x3)
    
    x32 = _conv_block(x3, 32, (1, 1), strides=(1, 1))
    x32 = UpSampling2D(2)(x32)#//16
    x2 = tf.keras.layers.add([x2, x32])#32 + 16 
    x32 = mbv1_depthwise_conv_block(x2, 32, 1,1,block_id=252)
    x2 = tf.keras.layers.add([x2, x32])#32 + 16 
    x21 = _conv_block(x2, first_block_filters, (1, 1), strides=(1, 1))#16
    x21 = UpSampling2D(2)(x21)#//16 -> 8
    x1 = tf.keras.layers.add([x1, x21])#8
    x21 = mbv1_depthwise_conv_block(x1, first_block_filters, 1,1,block_id=251)
    x1 = tf.keras.layers.add([x1, x21])#8
    x21 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'valid')(x1)# 8 ->16

    #//8 branch
    x1 = _conv_block(x1, 192, (1, 1), (1, 1))
    x1 = tf.keras.layers.DepthwiseConv2D((3,3), strides=(1, 1), depth_multiplier=1, padding='same')(x1)
    x1 = tf.keras.layers.BatchNormalization(axis=channel_axis)(x1)
    x1 = Relu6(x1)
    x1 = tf.keras.layers.Conv2D(144, (1, 1), strides=(1, 1), padding='same')(x1)
    x1 = Relu6(x1)
    x1 = tf.keras.layers.Conv2D(num_anchors*(num_classes+5), (1,1), strides=1, padding="same",name = 'out_first')(x1)

    # //16 branch
    x21 = _conv_block(x21, 32, (1, 1), strides=(1, 1))#16
    x2 = tf.keras.layers.add([x2, x21])#16 
    x21 = mbv1_depthwise_conv_block(x2, 32, 1,1,block_id=250)
    x2 = tf.keras.layers.add([x2, x21])#16    
    x32 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'valid')(x2)# 16 ->32
    
    x2 = _conv_block(x2, 192, (1, 1), (1, 1))
    x2 = tf.keras.layers.DepthwiseConv2D((3,3), strides=(1, 1), depth_multiplier=1, padding='same')(x2)
    x2 = tf.keras.layers.BatchNormalization(axis=channel_axis)(x2)
    x2 = Relu6(x2)
    x2 = tf.keras.layers.Conv2D(144, (1, 1), strides=(1, 1), padding='same')(x2)
    x2 = Relu6(x2)
    x2 = tf.keras.layers.Conv2D(num_anchors*(num_classes+5), (1,1), strides=1, padding="same",name = 'out_third')(x2)

    # //32 branch
    x32 = _conv_block(x32, 32, (1, 1), strides=(1, 1))#32
    x3 = _conv_block(x3, 32, (1, 1), strides=(1, 1))#32
    x3 = tf.keras.layers.add([x3, x32])#32
    x32 = mbv1_depthwise_conv_block(x3, 32, 1,1,block_id=249)
    x3 = tf.keras.layers.add([x3, x32])#32
    
    x3 = _conv_block(x3, 192, (1, 1), (1, 1))
    x3 = tf.keras.layers.DepthwiseConv2D((3,3), strides=(1, 1), depth_multiplier=1, padding='same')(x3)
    x3 = tf.keras.layers.BatchNormalization(axis=channel_axis)(x3)
    x3 = Relu6(x3)
    x3 = tf.keras.layers.Conv2D(144, (1, 1), strides=(1, 1), padding='same')(x3)
    x3 = Relu6(x3)
    x3 = tf.keras.layers.Conv2D(num_anchors*(num_classes+5), (1,1), strides=1, padding="same",name = 'out_second')(x3)
    return Model(body.input, [x3,x2,x1]) 
    

def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])
    
    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    try:
        grid = K.cast(grid, K.dtype(feats))
    except:
        grid = K.cast(grid, K.dtype(K.constant(feats)))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores

def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False, obj_scale=1,noobj_scale=1):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    if(num_layers > 1):
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5],[0,1,2]]
        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
        grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
        m = K.shape(yolo_outputs[0])[0] # batch size, tensor
        mf = K.cast(m, K.dtype(yolo_outputs[0]))
    else:
        anchor_mask = [[0,1,2]]
        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
        grid_shapes = K.cast(K.shape(yolo_outputs[0])[1:3], K.dtype(y_true[0]))
        m = K.shape(yolo_outputs[0])[0] # batch size, tensor
        mf = K.cast(m, K.dtype(yolo_outputs[0]))

    
    loss = 0
    
    
    for l in range(num_layers):
        #tf.print("\r\r****** l anchors:,grid_shapes",l,anchors[anchor_mask[l][0]])
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            try:
                tf.print("@@@train:",np.max(best_iou.numpy()),np.average(best_iou.numpy()))
            except:
                print()
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        
        _, ignore_mask = tf.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        confidence_loss = obj_scale * object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            noobj_scale * (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            tf.print("\n-----loss:",l,xy_loss, wh_loss, confidence_loss, class_loss, K.max(K.sigmoid(raw_pred[...,4:5])))
            #tf.print('\n---\n')
    return loss

from utils import yolo_cfg, get_random_data,mbv1_pretrained_weights
if __name__ == '__main__':
    
    def find_weights(net_w,net_h,net_alpha):
        net_size = min(net_w,net_h)
        for weight in mbv1_pretrained_weights:
            if [net_size,net_size,net_alpha] == weight:
                return True, './weights/mbv1_%s_%d_%d.h5'%(str(net_alpha),net_size,net_size)
        
        return False,None

    h,w = 128,160
    input_shape = (h,w,3)
    alpha = 0.5
    weight_load, weight = find_weights(w,h,alpha)
    print("\nfind weight:%s\n"%weight)
    m = tiny_yolo_res_body(input_shape,3,1,alpha,weights_path=weight,weight_load=True)
    m.summary()

    m.save('demo_%d_%d.h5'%(w,h))

    converter = tf.lite.TFLiteConverter.from_keras_model(m)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supportes_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    cfg = yolo_cfg()
    annotation_path = 'train_data.txt'
    cfg.width = w
    cfg.height = h
    with open(annotation_path) as f:
        annotation_lines = f.readlines()
    annotation_lines = annotation_lines[0:500]
    def representative_data_gen():
        '''data generator for fit_generator'''
        n = len(annotation_lines)
        i = 0
        input_shape = (cfg.height,cfg.width)
        
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
            input_value = input_value.reshape(1,cfg.height,cfg.width,3)
            yield [input_value]

    converter.representative_dataset = representative_data_gen
    tflite_model_quant = converter.convert() 
    m_path = 'demo_%d_%d.tflite'%(w,h)
    with open(m_path,'wb') as f:
        f.write(tflite_model_quant)
        f.close()

    print('save %s done'%m_path)

