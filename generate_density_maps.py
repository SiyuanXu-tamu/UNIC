import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import cv2
import json
from matplotlib import cm as CM

#this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    # pts = np.array(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))
    pts = np.array(list(zip(np.nonzero(gt)[1].ravel(), np.nonzero(gt)[0].ravel())))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print ('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))//2.//2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print ('done.')
    return density


##now generate the RSOC_building ground truth
RSOC_train = os.path.join('../dataset/new_building/building_256/train_data/','images')
RSOC_test = os.path.join('../dataset/new_building/building_256/test_data/','images')


print(RSOC_train, RSOC_train)
path_sets = [RSOC_train,RSOC_train]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
print(len(img_paths))

train_list = []
test_list = []
for img_path in glob.glob(os.path.join(RSOC_train, '*.jpg')):
    train_list.append(img_path)
for img_path in glob.glob(os.path.join(RSOC_test, '*.jpg')):
    test_list.append(img_path)
        
for img_path in img_paths:
    gd_path = img_path.replace('jpg', 'npy')
    
    gt = np.load(gd_path, allow_pickle=True).astype(np.float32)
    gt = gt[:, :2]
    
    
    img= plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1

    # k = gaussian_filter_density(k)
    k = gaussian_filter(k, 4)
    groundtruth = np.asarray(k)
    print(groundtruth.shape)
    with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
            hf['density'] = k
