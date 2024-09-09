import matplotlib.pyplot as plt
import numpy as np


'''
total_epoc = 200
warmEpoch = 20
maxLR = 0.0025
min_learn_rate = 1e-6
cosHalfCycleEpoch = (total_epoc-warmEpoch) // 2
lr_list = []
for epoc in range(total_epoc):
    if epoc < warmEpoch:
        lr = (maxLR - min_learn_rate) / (warmEpoch + 1) * (epoc + 1) + min_learn_rate
    else:
        epoc -= warmEpoch
        ndx = epoc % cosHalfCycleEpoch
        k = cosHalfCycleEpoch
        amp = (1 + (epoc // cosHalfCycleEpoch) * 1.5)
       
        lr = maxLR * (1 + np.math.cos(np.pi / k * ndx)) / 2 /amp + min_learn_rate
    lr_list.append(lr)

plt.plot(np.array(lr_list))
plt.grid()
plt.show()
'''
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('-lr', help='trained tflite', default=r'yolo3_192_192_0.25_nano_final.tflite',type=str)
    parser.add_argument('-train', help='test image', default=r'./person.jpeg',type=str)
    parser.add_argument('-val', help='test image', default=r'./192_192_anchors.txt',type=str)
    args, unknown = parser.parse_known_args()

    lr = np.load(args.lr)
    train_loss = np.load(args.train)
    val_loss = np.load(args.val)

    plt.plot(lr)
    plt.grid()
    plt.show()

    val_loss = val_loss[10:]
    train_loss = train_loss[10:]
    lr = lr[10:]*1000 + 8
    plt.plot(val_loss)
    #plt.plot(train_loss)
    plt.plot(lr)
    plt.grid()
    plt.show()