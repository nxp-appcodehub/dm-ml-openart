## Copyright 2023 NXP  ##
import tf,os,image,ujson,cmath,nxp_module,struct
from ulab import numpy as np



threadhold = 0.90
det_model = tf.load("/sd/yolo_face_det.tflite",load_to_fb=True)
mobilenet = tf.load("/sd/mfn_drop_320_best_quant.tflite")
db_file = "/sd/mobilefacenet/db.json"
db_list = []
arr_list = []
db_count = 0
db_base_buffer = 0
def clear_faces():
    global db_count
    global db_list

    for dict in db_list:
        idx = dict['idx']
        fname = "/sd/mobilefacenet/%d.bmp"%idx
        os.remove(fname)
        fname = "/sd/mobilefacenet/%d.npy"%idx
        os.remove(fname)
    try:
        os.remove(db_file)
    except:
        print("no db")

    db_count = 0
    db_list.clear()

def load_db():
    global db_count
    global db_list
    global arr_list

    try:
        os.stat('/sd/mobilefacenet')
    except:
        os.mkdir('/sd/mobilefacenet')
    try:
        f = open(db_file,'r')
        db_str = f.read()
        db_list = ujson.loads(db_str)
        f.close()
    except:
        print("no face found")
    db_count = len(db_list)

    for db in db_list:
        vector = np.load(db['vector'])
        arr_list.append(vector)

    return len(db_list)

def get_faces_cnt():
    return len(db_list)

def save_db():
    global db_list
    global db_base_buffer
    '''
    try:
        os.remove(db_file)
    except:
        print("no db file")
    '''
    db_string = ujson.dumps(db_list)
    f = open(db_file,'w')
    f.write(db_string)
    f.close()


def save_face(arrs, img):
    global db_count
    global db_list
    global arr_list

    dict = {"idx":0,"file":"","vector":1}
    fname = '/sd/mobilefacenet/%d'%db_count+'.bmp'
    arrs = np.array(struct.unpack("128f",arrs))
    
    dict['idx'] = db_count
    dict['file'] = fname
    dict['vector'] = '/sd/mobilefacenet/%d'%db_count+'.npy'
    img.save(fname)
    np.save(dict['vector'],arrs)
    db_count = db_count +1
    db_list.append(dict)
    arr_list.append(arrs)
    save_db()
    print("save image: %s"%fname)

#calculate face feature
def ModelCalculateArray(img):
    global mobilenet
    
    out = tf.invoke(mobilenet, img)
    array = out.output()[0][0]
    return array

def findFaceinList(arr, angle_threadhold,angle_sensitive):
    global db_list
    global arr_list

    if len(arr_list) == 0:
        return -1,999
    min = 999.0
    i = 0
    index = -1
    
    for save_arr in arr_list:
        tensor = save_arr.tobytes()

        angle = nxp_module.compare_tensor_angle(base=arr,tensor=tensor,len=128)
        if angle < min:
            min = angle
            index = i
        i += 1
    print(min)
    if(min > angle_threadhold + angle_sensitive):
        index = -1
        min = 999

    return index,min

#find face in preview image
def find_faces(img):
    global det_model
    global threadhold

    
    objs = tf.detect(det_model,img,bgr=0)
    obj_list = []
    for obj in objs:
        x1,y1,x2,y2,label,scores = obj
        if (scores > threadhold):
            w = x2- x1
            h = y2 - y1
            x1 = int(x1*img.width() )
            y1 = int(y1*img.height())
            w = int(w*img.width() )
            h = int(h*img.height() )
            obj_list.append((x1,y1,w,h))

    return obj_list

