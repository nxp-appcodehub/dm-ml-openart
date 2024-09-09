from train import model_train
from utils import yolo_cfg
from kmeans import YOLO_Kmeans

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
if __name__ == '__main__':

    cfg = yolo_cfg()

    for size in size_supported:
        cfg.height,cfg.width = size
        cfg.cluster_anchor = '%d_%d_anchors.txt'%(cfg.width,cfg.height)
        kmeans = YOLO_Kmeans(cfg)
        kmeans.txt2clusters()

        for i in range(len(mbv1_alpha)):
            cfg.alpha = mbv1_alpha[i]
            model_train(cfg)


    
    