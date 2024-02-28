import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import scipy
import cv2
import json
from matplotlib import cm as CM
import torch

building_train = os.path.join('../dataset/building_counting/RSOC_building/building/train_data/','images')
building_test = os.path.join('../dataset/building_counting/RSOC_building/building/test_data/','images')

gt_path = '../dataset/building_counting/RSOC_building/building_bay_64/train_data/ground_truth/'

path_sets = [building_train]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
print(len(img_paths))

for img_path in img_paths:
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
    img= plt.imread(img_path)
    
    gt = mat['center'][0,0]
    
    origin_size = img.shape[0]
    target_size = 64
    ratio = int(origin_size/target_size)
    
    #print(img_path)
    
    
    if len(gt) > 0:
        prior_prob = np.zeros((len(gt), img.shape[0] // ratio,img.shape[1] // ratio))
    
    
        for i in range(0,len(gt)):
            if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
                k = np.zeros((img.shape[0],img.shape[1]))
                k[int(gt[i][1]),int(gt[i][0])]=1
                k = gaussian_filter(k, 8, mode='constant')
                k = np.asarray(k)
                k_ = cv2.resize(k, (k.shape[1] // ratio, k.shape[0] // ratio), interpolation=cv2.INTER_CUBIC) * ratio * ratio
                
                prior_prob[i] = k_
    else:
        prior_prob = np.zeros((1, img.shape[0] // ratio,img.shape[1] // ratio))
    
    name = img_path.split('/')
    #print(img_path)
    print(name[-1][:-4])
    #print(gt_path+name[-1][:-4]+'.h5')
    
    with h5py.File(gt_path+name[-1][:-4]+'.h5', 'w') as hf:
            hf['prior_prob'] = prior_prob
            
    