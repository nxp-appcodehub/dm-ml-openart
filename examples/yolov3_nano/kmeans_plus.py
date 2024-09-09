import numpy as np
from utils import yolo_cfg
from PIL import Image 
class YOLO_Kmeans:

    def __init__(self,cfg):
        
        self.cluster_number = cfg.cluster_number
        self.filename = 'train.txt'
        self.anchor_fname = cfg.cluster_anchor
        self.target_w = cfg.width
        self.target_h = cfg.height
    def cas_iou(self,box,cluster):
        x = np.minimum(cluster[:,0],box[0])
        y = np.minimum(cluster[:,1],box[1])

        intersection = x * y
        area1 = box[0] * box[1]

        area2 = cluster[:,0] * cluster[:,1]
        iou = intersection / (area1 + area2 -intersection)

        return iou

    def avg_iou(self,box,cluster):
        return np.mean([np.max(self.cas_iou(box[i],cluster)) for i in range(box.shape[0])])

    def iou(self, boxes, clusters,k):  # 1 box -> k clusters
        n = boxes.shape[0]
        #k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result
    
    def avg_iou_heads(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters,self.cluster_number), axis=1)])
        print(clusters.shape)
        clusters = clusters.reshape(int(self.cluster_number/3),3,2)
        accs = []
        for cluster in clusters:
            accs.append(np.mean([np.max(self.iou(boxes, cluster,3), axis=1)]))

        return accuracy,accs
    
    def bboxesOverRation(self,bboxesA,bboxesB):

        bboxesA = np.array(bboxesA.astype('float'))
        bboxesB = np.array(bboxesB.astype('float'))
        M = bboxesA.shape[0]
        N = bboxesB.shape[0]
        
        areasA = bboxesA[:,2]*bboxesA[:,3]
        areasB = bboxesB[:,2]*bboxesB[:,3]
        
        xA = bboxesA[:,0]+bboxesA[:,2]
        yA = bboxesA[:,1]+bboxesA[:,3]
        xyA = np.stack([xA,yA]).transpose()
        xyxyA = np.concatenate((bboxesA[:,:2],xyA),axis=1)
        
        xB = bboxesB[:,0] +bboxesB[:,2]
        yB = bboxesB[:,1]+bboxesB[:,3]
        xyB = np.stack([xB,yB]).transpose()
        xyxyB = np.concatenate((bboxesB[:,:2],xyB),axis=1)
        
        iouRatio = np.zeros((M,N))
        for i in range(M):
            for j in range(N):
                x1 = max(xyxyA[i,0],xyxyB[j,0])
                x2 = min(xyxyA[i,2],xyxyB[j,2])
                y1 = max(xyxyA[i,1],xyxyB[j,1])
                y2 = min(xyxyA[i,3],xyxyB[j,3])
                Intersection = max(0,(x2-x1))*max(0,(y2-y1))
                Union = areasA[i]+areasB[j]-Intersection
                iouRatio[i,j] = Intersection/Union
        return iouRatio


    


    def estimateAnchorBoxes(self,trainingData,numAnchors=9):
        
        numsObver = trainingData.shape[0]
        xyArray = np.zeros((numsObver,2))
        trainingData[:,0:2] = xyArray
        assert(numsObver>=numAnchors)
        
        # kmeans++
        # init 
        centroids = [] 
        centroid_index = np.random.choice(numsObver, 1)
        centroids.append(trainingData[centroid_index])
        while len(centroids)<numAnchors:
            minDistList = []
            for box in trainingData:
                box = box.reshape((-1,4))
                minDist = 1
                for centroid in centroids:
                    centroid = centroid.reshape((-1,4))
                    ratio = (1-self.bboxesOverRation(box,centroid)).item()
                    if ratio<minDist:
                        minDist = ratio
                minDistList.append(minDist)
                
            sumDist = np.sum(minDistList)
            prob = minDistList/sumDist 
            idx = np.random.choice(numsObver,1,replace=True,p=prob)
            centroids.append(trainingData[idx])
            
        maxIterTimes = 100
        iter_times = 0
        while True:
            minDistList = []
            minDistList_ind = []
            for box in trainingData:
                box = box.reshape((-1,4))
                minDist = 1
                box_belong = 0
                for i,centroid in enumerate(centroids):
                    centroid = centroid.reshape((-1,4))
                    ratio = (1-self.bboxesOverRation(box,centroid)).item()
                    if ratio<minDist:
                        minDist = ratio
                        box_belong = i
                minDistList.append(minDist)
                minDistList_ind.append(box_belong)
            centroids_avg = []
            for _ in range(numAnchors):
                centroids_avg.append([])
            for i,anchor_id in enumerate(minDistList_ind):
                centroids_avg[anchor_id].append(trainingData[i])
            err = 0
            for i in range(numAnchors):
                if len(centroids_avg[i]):
                    temp = np.mean(centroids_avg[i],axis=0)
                    err +=  np.sqrt(np.sum(np.power(temp-centroids[i],2)))
                    centroids[i] = np.mean(centroids_avg[i],axis=0)
            iter_times+=1
            if iter_times>maxIterTimes or err==0:
                break
        anchorBoxes = np.array([x[2:] for x in centroids])
        meanIoU = 1-np.mean(minDistList)
        anchorBoxes = anchorBoxes[np.argsort(anchorBoxes[:,0])]
        print('acc:{:.2f}%'.format(self.avg_iou(trainingData[:,2:],anchorBoxes) * 100))
        anchorBoxes = anchorBoxes.astype('int32')
        return anchorBoxes,meanIoU

    def result2txt(self, data):
        f = open(self.anchor_fname, 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        f = open(self.filename, 'r')
        dataSet = []
        count = 0
        total_files=0
        for line in f:
            infos = line.split("$")
            length = len(infos)
            
            img = Image.open(infos[0])
            img_w,img_h = img.size
            scale = min(self.target_w / img_w,self.target_h / img_h)
            w_scale = self.target_w / img_w
            h_scale = self.target_h / img_h
            for i in range(1, length):
                if int(infos[i].split(",")[0]) >= int(infos[i].split(",")[2]) or int(infos[i].split(",")[1]) >= int(infos[i].split(",")[3]):
                    print(line)
                    count += 1
                xmin = int(infos[i].split(",")[0]) 
                ymin = int(infos[i].split(",")[1]) 
                xmax = int(infos[i].split(",")[2]) 
                ymax = int(infos[i].split(",")[3])

                width = int(infos[i].split(",")[2]) - \
                    int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - \
                    int(infos[i].split(",")[1])
                
                width = int(width*scale)
                height = int(height*scale)

                if(height > self.target_h ) or (width > self.target_w):
                    print(img_w,img_h,infos[i].split(","))
                x = xmin + 0.5 * (xmax-xmin)
                y = ymin + 0.5 * (ymax - ymin)
                total_files = total_files +1
                dataSet.append([x,y,width, height])
                
        result = np.array(dataSet)
        f.close()
        print('box error count:%d in %d files'%(count,total_files))
        return result
    
    def calculate_anchors(self):
        data = self.txt2boxes()
        anchors, _ = self.estimateAnchorBoxes(data, numAnchors = self.cluster_number)
        avg_acc,accs = self.avg_iou_heads(data[..., 2:],anchors)
        print(anchors)
        print("Avg Accuracy: {:.2f}%".format(
            avg_acc * 100))
        for acc in accs:
            print(" Accuracy: {:.2f}%".format(
                acc * 100))
            
        f = open(self.anchor_fname, 'w')
        row = np.shape(anchors)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (anchors[i][0], anchors[i][1])
            else:
                x_y = ", %d,%d" % (anchors[i][0], anchors[i][1])
            f.write(x_y)
        f.close()


if __name__ == "__main__":
    cfg = yolo_cfg()
    kmeans = YOLO_Kmeans(cfg)
    kmeans.calculate_anchors()
