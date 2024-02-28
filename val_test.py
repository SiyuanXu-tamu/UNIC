import os
import numpy as np

import torch

import argparse

import inspect


#from  models.aspd_sr_duo_vae import ASRNet
#from  models.aspd_duo_test import New_bay_Net
from models.aspd_spatial_uq1 import New_bay_Net

from datasets.crowd_duo_vae import Crowd
from losses.bay_loss_new import Bay_Loss
from losses.post_prob_duo import Post_Prob



from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


import cv2
from matplotlib import pyplot as plt



from torchvision import datasets, transforms

from scipy.stats import norm


args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default=#'../dataset/building_counting/RSOC_building/building_bay/',
                                              '../dataset/building_counting/RSOC_building/building_bay_256/',
                        help='data directory')
    parser.add_argument('--save-dir', default='pre_density/',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--input_size', default=256, help='test image size')
    parser.add_argument('--downsample-ratio', type=int, default=2,
                        help='downsample ratio')
    parser.add_argument('--is-gray', type=bool, default=False,
                        help='whether the input image is gray')
    parser.add_argument('--pretrained_model_dir', default='pretrained_model/',
                        help='path to pretrained model')
    parser.add_argument('--use_sr', type=bool, default=False,
                        help='use_sr')
    parser.add_argument('--sigma', type=float, default=4.0,##8.0
                        help='sigma for likelihood')
    parser.add_argument('--use-background', type=bool, default=True,
                        help='whether to use background modelling')
    args = parser.parse_args()
    return args
    
def train_collate(batch):

    ##### img, sr_img, keypoints, prior_prob, st_size, grid_c, grid_sr, gridnum_sam_c

    transposed_batch = list(zip(*batch))
    
    images = torch.stack(transposed_batch[0], 0) ####img
    
    sr_images = torch.stack(transposed_batch[1], 0)### sr_img
    
    points = transposed_batch[2]  # keypoints, the number of points is not fixed, keep it as a list of tensor
    
    prior_prob = transposed_batch[3] ### prior_prob
    
    st_sizes = torch.FloatTensor(transposed_batch[4]) ###st_size  shortest size = min(w,h)
    
 
    grid_c = torch.stack(transposed_batch[5], 0)
    
    grid_sr = torch.stack(transposed_batch[6], 0)
    
    gridnum_sam_c = transposed_batch[7] ####gridnum_sam_c
    
    gd_count = transposed_batch[8]
    
    
    
    return images, sr_images, points, prior_prob, st_sizes, grid_c, grid_sr, gridnum_sam_c, gd_count


if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    
    
    datasets = Crowd(os.path.join(args.data_dir, 'test_data/images'), args.input_size, args.downsample_ratio, args.is_gray, 'val')
    dataloader = DataLoader(datasets, 
                            collate_fn=default_collate,
                            batch_size=1,
                            shuffle=False,
                            num_workers=8, 
                            pin_memory=False)



    #model =ASRNet(args.downsample_ratio)
    model =New_bay_Net(args.downsample_ratio, 256)
    device = torch.device('cuda')
    #
    #model.load_state_dict(torch.load(os.path.join(args.pretrained_model_dir, 'our_method_256_test.pth'), device))
    model = torch.load(os.path.join(args.pretrained_model_dir, 'fine_model.pt'))
    #ckeckpoint = torch.load(os.path.join(args.pretrained_model_dir, '1686_ckpt.tar'))
    #model.load_state_dict(ckeckpoint['model_state_dict'])
    model.to(device)
    model.eval()

    
    print(model)
    #print(inspect.getsource(model.forward))
    
    epoch_res = []
    
    sr_results = []
    c_results = []
    post_p = []
    pd = []
    gt = []
    post_prob = Post_Prob(args.sigma,
                                   args.input_size,
                                   args.downsample_ratio,
                                   1.0,
                                   args.use_background,
                                   device)
    sigma = []
    good = []
    sigma_l = []

    for inputs, count, name, cor_C, cor_HR in dataloader:
        #print('val input',inputs.shape)
        inputs = inputs.to(device)
        cor_C = cor_C.to(device)
        cor_HR = cor_HR.to(device)
        # inputs are images with different sizes
        assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
        with torch.set_grad_enabled(False):          
            if args.use_sr:  
                outputs, sr_imgs, f1, f2 = model(inputs, cor_C, cor_HR, args.use_sr)
                    
                sr_results.append(sr_imgs.data.cpu().numpy())
            else:
                outputs, f1, f2, out_sigma = model(inputs, cor_C, cor_HR, args.use_sr, 'test',0)
            
            outputs = outputs/10 #/4
        
               
            res = count[0].item() - torch.sum(outputs).item() 
                
            epoch_res.append(res)
            print(res, count[0].item(), torch.sum(outputs).item(),name)
            
            out_sigma = 3.5*out_sigma
            
            var = torch.sqrt(torch.sum(torch.pow(out_sigma,2))).data.cpu().numpy()
            #print(res, var)
            
            #print(outputs.data.cpu().numpy().shape)
                
            c_results.append(outputs.data.cpu().numpy())
            sigma.append(var)
            sigma_l.append(out_sigma.data.cpu().numpy())
            
            if np.abs(res)<1:
                good.append([len(c_results), name])
    
    epoch_res = np.array(epoch_res)
    mse = np.sqrt(np.mean(np.square(epoch_res)))
    mae = np.mean(np.abs(epoch_res))
        
    print(mse, mae)
    #print(good)
    
    sr_results = np.array(sr_results)
    c_results = np.array(c_results)
    #np.save('srr_image.npy', sr_results)
    np.save(os.path.join(args.save_dir, 'our_method_256_123_4_128.npy'), c_results)
    #sigma_l = np.array(sigma_l)
    #np.save(os.path.join(args.save_dir, 'our_method_256_128_sigma.npy'), sigma_l)
    
    
    sigma = np.array(sigma)
    
    
    fig = plt.figure(figsize=(10, 7))
    plt.plot(sigma, np.abs(epoch_res), 'bo')
    #plt.savefig('test_fig1.png')
    plt.close()
    
    
    
    n_test = epoch_res.shape[0]
    
    ratios = []
    x_axis = []
    
    
    
    
    nn = 500
    
    for i in range(0, nn+1):
        n = 0
        
        for j in range(n_test):
            
            gt = np.abs(epoch_res[j])/(sigma[j])
            
            if 2*norm.cdf(gt) -1 <= min(i/nn,1):
                n += 1
        ratio = n/n_test
        x_axis.append(min(i/nn,1))
        ratios.append(ratio)
    
    
    ratios = np.array(ratios)
    x_axis = np.array(x_axis)
    
    
    fig = plt.figure(figsize=(10, 7))
    plt.plot(x_axis, ratios, 'bo')
    #plt.savefig('test_fig2.png')
    plt.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
