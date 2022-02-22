'''
AUTHOR: AMARTYA MANNAVA

TODO use pretrained weights for deployment

TODO    # frcnn
        # attention model

'''

import  matplotlib.pyplot as mp
import numpy as np

def buildNET(path):
    return 0

if __name__ =='__main__':
    x = np.arange(0,20)

    y= np.array([52.34, 58.12 , 60.35, 60.35, 60.35,
                 65.87, 65.97, 66.03, 68.12, 73.59,
                 75.98, 78.53,80.12, 80.12, 80.12,
                 84.53, 88.56, 90.43, 92.5, 94.6])
    mp.ylabel('Accuracy')
    mp.xlabel('Epochs')
    mp.plot(x,y)
    mp.ylim(ymin=0)
    mp.show()
