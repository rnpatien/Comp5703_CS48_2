import sys
import pickle
import glob
import os
import imageio
import numpy as np

input_path='./data/cifar-10-batches-py/'
output_path='./data/cifar-10-imbalance/'

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == '__main__':
    meta=unpickle(input_path+ 'batches.meta')
    img_cnts=[0]* 10

    if not os.path.exists(output_path):
        os.mkdir(output_path)
        tandt=['train','test']
        for tt in tandt:
            toutput_path=output_path+tt+'/'
            os.mkdir(toutput_path)
            for ii in range(10):
                if not os.path.exists(toutput_path+ '/' + str(ii)):
                    os.mkdir(output_path + tt +'/' + str(ii))

    wrkdic=unpickle(input_path+ 'test_batch' )
    labs=wrkdic[b'labels']
    imgs=wrkdic[b'data']
    for jj,lab in enumerate(labs):
        wrkdir= output_path+ 'test/' + str(lab)
        np.save(wrkdir + '/' + str(img_cnts[lab]),imgs[jj])
        img_cnts[lab] +=1

    for ii in range(1,5):
        wrkdic=unpickle(input_path+ 'data_batch_' + str(ii))
        labs=wrkdic[b'labels']
        imgs=wrkdic[b'data']
        for jj,lab in enumerate(labs):
            wrkdir= output_path+ 'train/' + str(lab)
            np.save(wrkdir + '/' + str(img_cnts[lab]),imgs[jj])
            img_cnts[lab] +=1


            
    aa=1