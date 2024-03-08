from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import argparse


def cal_new_size(im_h, im_w, min_size, max_size):###limit the min and max size of img
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w*ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def find_dis(point):
    square = np.sum(point*points, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis

def generate_data(im_path, ratio):
    im = Image.open(im_path)
    im = im.resize([im.size[0] // ratio, im.size[1] // ratio])
    
    im_w, im_h = im.size
    #mat_path = im_path.replace('.jpg', '_ann.mat')
    #points = loadmat(im_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))['image_info'][0, 0][0, 0][0].astype(np.float32) ###['center'][0,0]
    points = loadmat(im_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))['center'][0,0].astype(np.float32)
    #key = points.keys()
    points = points/ratio
    
    #points = loadmat(mat_path)['annPoints'].astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--origin-dir', default='/home/teddy/UCF-QNRF_ECCV18',
                        help='original data directory')
    parser.add_argument('--data-dir', default='/home/teddy/UCF-Train-Val-Test',
                        help='processed data directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    min_size = 256
    max_size = 2048
    
    origin_dir = '../dataset/building_counting/RSOC_building/building'
    data_dir = '../dataset/building_counting/RSOC_building/building_bay_256'
    

    
    ratio = int(512/256)

    

    for phase in ['train_data', 'test_data']:
        if True:
            sub_dir = os.path.join(origin_dir, phase)
            sub_dir_im = os.path.join(sub_dir, 'images')
            
            save_dir = os.path.join(data_dir, phase)
            sub_save_dir = os.path.join(save_dir, 'images')
            
            for im_path in glob(os.path.join(sub_dir_im, '*.jpg')):
                        #im_path = os.path.join(sub_dir, i.strip())
                name = os.path.basename(im_path)
                print(name)
                im, points = generate_data(im_path, ratio)
                        
                dis = find_dis(points)
                points = np.concatenate((points, dis), axis=1)
                
                im_save_path = os.path.join(sub_save_dir, name)
                im.save(im_save_path)
                gd_save_path = im_save_path.replace('jpg', 'npy')
                np.save(gd_save_path, points)