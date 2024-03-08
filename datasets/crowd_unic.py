from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np

import h5py
import cv2



def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w

 
def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
    return inner_area
    
    
    
def get_coordinates(res_original, k_HR: int):
    '''
    Generate coordinate matrices
    Args:
        k_HR (int): high resolution data reduction ratio
        k_LR (int): low resolution data reduction ratio
    Returns:
        d_HR (torch.tensor): high resolution data sample coordinate matrix
    '''
    res_original = res_original
    #lower, upper = -0.5, 0.5 #0.01, 0.99
    lower, upper = -0.99, 0.99 
    
    #lower = -1 + 2/res_original/2
    #upper = 1 - 2/res_original/2
    x = np.linspace(lower, upper, res_original)#, endpoint=True
    y = np.linspace(lower, upper, res_original)
    #x = np.linspace(lower, upper, res_original, endpoint=False)#, endpoint=True
    #y = np.linspace(lower, upper, res_original, endpoint=False)
    xx, yy = np.meshgrid(x, y)

    dx_HR, dy_HR = [], []
    for i in range(0, res_original, k_HR):
        tmp_x, tmp_y = [], []
        for j in range(0, res_original, k_HR):
            tmp_x.append(xx[i][j])
            tmp_y.append(yy[i][j])
        dx_HR.append(tmp_x)
        dy_HR.append(tmp_y)
    
    d_HR = np.array([dx_HR, dy_HR])
    del dx_HR, dy_HR, xx, yy
    

    return torch.tensor(np.transpose(np.reshape(d_HR, (d_HR.shape[0], d_HR.shape[1]*d_HR.shape[2])), [1,0]), dtype=torch.float)



class Crowd(data.Dataset):
    def __init__(self, root_path, crop_size,
                 downsample_ratio,is_gray=False,
                 method='train'):
    
        self.root_path = root_path######path to images
        
        #self.use_bg = use_bg
        
        
        ######find the image path
        if True:
        #if method=='train':
            self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))####img list
        else:
            origin_root_path = '/mnt/shared-scratch/siyuanxu/Counting/dataset/building_counting/RSOC_building/building_bay/test_data/images'
            self.im_list = sorted(glob(os.path.join(origin_root_path, '*.png')))####img list
            
            
            
            
        if method not in ['train', 'val']:
            raise Exception("not implement")
            
            
        
        
        self.method = method#### train or val
        
        print(self.method)
        
        
        self.c_size = crop_size#######do crop, crop size
        self.d_ratio = downsample_ratio######down sample rate in the network 8
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio####after down size
        
        ### initial spatial coordinates

        
        
        self.density_size = self.c_size // self.d_ratio #*2
        
        
        self.cor_C = get_coordinates(self.density_size, 1)
        
        ##counting
        self.gridnum_list_c = [i for i in range(self.density_size*self.density_size)]
        
        if self.density_size == 32:
            self.cor_sample_size_c = 32
        elif self.density_size == 64:
            self.cor_sample_size_c = 50
        elif self.density_size == 128:
            self.cor_sample_size_c = 64
        elif self.density_size == 8:
            self.cor_sample_size_c = 8
        else:
            self.cor_sample_size_c = 150
        self.n_query_pts_c = self.cor_sample_size_c*self.cor_sample_size_c####select 6*6 from 8*8
        self.gridnum_sam_c = np.random.choice(self.gridnum_list_c, size=self.n_query_pts_c, replace=False, p=None)
        
        
        
        
        ####normalization for images
        if is_gray:#####normalize
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item] 
        gd_path = img_path.replace('jpg', 'npy')  #####this path is for distance and keypoints
        
        
        ####read img and sr img
        img = Image.open(img_path).convert('RGB')#### read img

        
        if self.method == 'train':
        
            ###read points and dis
            keypoints = np.load(gd_path)
            
            ##density
            den_path = img_path.replace('building_bay', 'building').replace('.jpg', '.h5').replace('images', 'ground_truth')
            gt_file = h5py.File(den_path)
            den_target = np.asarray(gt_file['density'])
            #den_target = cv2.resize(den_target, (self.density_size, self.density_size), interpolation=cv2.INTER_CUBIC) *(2*self.d_ratio)*(2*self.d_ratio)
            den_target = cv2.resize(den_target, (self.density_size, self.density_size), interpolation=cv2.INTER_CUBIC) *(self.d_ratio)*(self.d_ratio)

            
            ###read prior
            prior_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
            #print(prior_path)
            prior_prob = h5py.File(prior_path)['prior_prob']
            prior_prob = np.array(prior_prob)
            #print(prior_prob.shape)
            return self.train_transform(img, keypoints, prior_prob, den_target)
        elif self.method == 'val':
            
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            #print(name)
            return img, len(keypoints), name, self.cor_C

    def train_transform(self, img, keypoints, prior_prob, den_target):
        """random crop image patch and find people in it"""
        
        
        ###do cropping
        
        wd, ht = img.size#### w h
        st_size = min(wd, ht)#### short size
        assert st_size >= self.c_size
        assert len(keypoints) > 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)###crop
        
        gd_count = keypoints.shape[0]
        
        #### compute the target and new coordinate after cropping
        nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)#### dis min 4.0 max 128.0

        points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
        points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
        bbox = np.concatenate((points_left_up, points_right_down), axis=1)
        inner_area = cal_innner_area(j, i, j+w, i+h, bbox)
        origin_area = nearest_dis * nearest_dis
        ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
        mask = (ratio >= 0.3)###0.3
        
        
        ####delete some points
        target = ratio[mask]
        keypoints = keypoints[mask]
        keypoints = keypoints[:, :2] - [j, i]  # change coodinate
        
        #if self.use_bg:
        if True:
            mask_prior = np.full(mask.shape[0] +1, True)
            mask_prior[:-1] = mask
            prior_prob = prior_prob[mask_prior]
        #else:
        #    prior_prob = prior_prob[mask]
        
        den_target = np.reshape(den_target, [1, -1])
        #print(prior_prob.shape, den_target.shape)
        prior_prob = np.concatenate((prior_prob, den_target), axis = 0)
        
        
        ######do flipping
        
        if len(keypoints) > 0:
            #if False:
            if random.random() > 0.5:
                img = F.hflip(img)###img
                keypoints[:, 0] = w - keypoints[:, 0]### location of head
                ###prior prob
                #if self.use_bg:
                prior_prob = prior_prob.reshape(len(keypoints)+2,self.density_size,self.density_size)
                prior_prob = np.flip(prior_prob, axis = 2)
                prior_prob = prior_prob.reshape(len(keypoints)+2, -1)
            
                
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                sr_img = F.hflip(sr_img)
                

        
        #####c
        #self.gridnum_sam_c = np.random.choice(self.gridnum_list_c, size=self.n_query_pts_c, replace=False, p=None)
        self.gridnum_sam_c = np.array(self.gridnum_list_c)
        
        prior_prob = prior_prob[:, self.gridnum_sam_c] ###[38, 64] -> [38, 36]
        #prior_prob = np.sum(prior_prob, axis = 1) ###[38, 36] -> [38]
        
        grid_c = self.cor_C[self.gridnum_sam_c, :] ### [8*8 2] -> [6*6, 2]  
        #grid_c =  self.cor_C2    
        
        #print(prior_prob.shape, target.shape)
        prior_prob[:-2] = prior_prob[:-2]*target.reshape([-1,1])
        #print(prior_prob[0])
        
        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), \
               torch.from_numpy(prior_prob.copy()).float(), st_size, grid_c, torch.from_numpy(self.gridnum_sam_c.copy()).float(), gd_count
