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

from scipy.special import softmax

building_train = os.path.join('../dataset/new_building/building_256/train_data/','images')
building_test = os.path.join('../dataset/new_building/building_256/test_data/','images')



gt_path = '../dataset/new_building/building_256/train_data/bayesian_prior/'



path_sets = [building_train]


down_sample_rate = 8  ### then the size of density map is target_size/down_sample_rate
sigma = 4


use_bg = True  ###whether use background
bg_ratio = 1.0
input_size = 256  ###image_size of building_train_path

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
print(len(img_paths))


for img_path in img_paths:
    img= plt.imread(img_path)
   
    gd_path = img_path.replace('jpg', 'npy')
    
    gt = np.load(gd_path, allow_pickle=True).astype(np.float32)
    gt = gt[:, :2]

    
    #origin_size = img.shape[0]
    
    origin_size = img.shape[0]
    target_size = 256   #### size after manual operation  512-64-8
    ratio = int(origin_size/target_size)
    
    gt = gt/ratio
    print('ratio', ratio)
    
    down_sample_step = down_sample_rate
    
    #print(img_path)
    
    
    if len(gt) > 0:
 
        if down_sample_step == 1:
            cood = np.arange(0, target_size, step=down_sample_step, dtype=np.float32)
        else:
            cood = np.arange(0, target_size, step=down_sample_step, dtype=np.float32) + down_sample_step / 2
        cood = cood[None, :]
        
        x = gt[:, 0][:, None]
        y = gt[:, 1][:, None]
        
        x_dis = -2 * np.matmul(x, cood) + x * x + cood * cood
        y_dis = -2 * np.matmul(y, cood) + y * y + cood * cood
        
        x_dis = np.expand_dims(x_dis, 1)
        y_dis = np.expand_dims(y_dis, 2)
        
        dis = x_dis + y_dis  #### 38 8 8
        
        dis = -dis / (2.0 * sigma ** 2)
        
        dis = dis.reshape(len(gt), -1)
        
        prior_prob = softmax(dis, axis = 1)
        

        
        print(np.sum(prior_prob[0]))
    
    else:
        r = target_size // down_sample_step
        prior_prob = np.zeros((1, r*r))
        
        
    if use_bg:
        if down_sample_step == 1:
            cood = np.arange(0, target_size, step=down_sample_step, dtype=np.float32)
        else:
            cood = np.arange(0, target_size, step=down_sample_step, dtype=np.float32) + down_sample_step / 2
        cood = cood[None, :]
        
        x = gt[:, 0][:, None]
        y = gt[:, 1][:, None]
        
        x_dis = -2 * np.matmul(x, cood) + x * x + cood * cood
        y_dis = -2 * np.matmul(y, cood) + y * y + cood * cood
        
        x_dis = np.expand_dims(x_dis, 1)
        y_dis = np.expand_dims(y_dis, 2)
        
        dis = x_dis + y_dis  #### 38 8 8
        dis = dis.reshape(len(gt), -1)
        
        min_dis =np.clip(np.min(dis, axis=0, keepdims=True)[0], a_min=0.0, a_max = None)
        #print(min_dis)
        
        bg_dis = (target_size * bg_ratio) ** 2 / (min_dis + 1e-5)
        #print(bg_dis)
        
        bg_dis = -bg_dis / (2.0 * sigma ** 2)
        #print(bg_dis)
        bg_map = np.exp(bg_dis) / sigma / 2.5
        #print(bg_map)
        #print(np.sum(bg_map), np.max(bg_map), np.min(bg_map))
        #print(prior_prob.shape)
        #np.save('bg_map', bg_map)
        bg_map = bg_map.reshape(1,-1)
        prior_prob = np.concatenate((prior_prob, bg_map), axis = 0)
        print(prior_prob.shape)
        
        
        
        
        
    
    
    name = img_path.split('/')
    #print(img_path)
    print(name[-1][:-4])
    #print(gt_path+name[-1][:-4]+'.h5')
    
    with h5py.File(gt_path+name[-1][:-4]+'.h5', 'w') as hf:
            hf['prior_prob'] = prior_prob

