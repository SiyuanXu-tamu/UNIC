from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.aspd_spatial_uq1 import New_bay_Net

from datasets.crowd_unic import Crowd
from losses.bay_loss_new import Bay_Loss
from losses.post_prob_duo import Post_Prob


import random

from torch.optim import lr_scheduler

import cv2
from matplotlib import pyplot as plt



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    
def get_parameters_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num


def train_collate(batch):

    ##### img, sr_img, keypoints, prior_prob, st_size, grid_c, grid_sr, gridnum_sam_c

    transposed_batch = list(zip(*batch))
    
    images = torch.stack(transposed_batch[0], 0) ####img
    
    
    points = transposed_batch[1]  # keypoints, the number of points is not fixed, keep it as a list of tensor
    
    prior_prob = transposed_batch[2] ### prior_prob
    
    st_sizes = torch.FloatTensor(transposed_batch[3]) ###st_size  shortest size = min(w,h)
    
 
    grid_c = torch.stack(transposed_batch[4], 0)
    
    
    gridnum_sam_c = transposed_batch[5] ####gridnum_sam_c
    
    gd_count = transposed_batch[6]
    
    
    
    return images, points, prior_prob, st_sizes, grid_c, gridnum_sam_c, gd_count


class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        
        ###Initial
        
        
        
        
        using_method = 'bay_vae'
        args = self.args
        

        
        
        logging.info('using seed {}'.format(args.seed))
        logging.info('using method {}'.format(using_method))
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        
        ###initial building dataset
        self.datasets = {x: Crowd((os.path.join(args.data_dir, 'train_data/images') if x == 'train' else os.path.join(args.data_dir, 'test_data/images')),
                                  args.crop_size,
                                  args.downsample_ratio,
                                  args.is_gray, x) for x in ['train', 'val']}

        
        g = torch.Generator()
        g.manual_seed(args.seed)
        
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                          if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers*self.device_count,
                                          pin_memory=(True if x == 'train' else False),
                                          worker_init_fn=(seed_worker if x == 'train' else None),
                                          generator=(g if x == 'train' else None),
                                          )
                            for x in ['train', 'val']}
        
        
        #####initial model
        

        self.model =New_bay_Net(self.downsample_ratio, args.crop_size)
        #self.model =ASRNet()
        self.model.to(self.device)
        
        self.use_sr = args.use_sr
        
        total_num, trainable_num = get_parameters_number(self.model)
        print(total_num, trainable_num)
        

        
        
        #####initial optimizer, selct parameters' learning rate
        

        
        #self.optimizer = optim.Adam(training_params)
        c_params = list(map(id, self.model.cc_decoder.last2.parameters()))
        b_params = filter(lambda p: id(p) not in c_params, self.model.parameters())
        
        
        self.optimizer1 = optim.Adam(b_params, lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer2 = optim.Adam(self.model.cc_decoder.last2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        
        #self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = lr_scheduler.StepLR(self.optimizer1, step_size = 3000, gamma = 1)

  
        self.start_epoch = 0
        self.epoch = self.start_epoch
        ######whether to load pre-trained model
        
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer1.load_state_dict(checkpoint['optimizer_state_dict1'])
                self.optimizer2.load_state_dict(checkpoint['optimizer_state_dict2'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))
            elif suf == 'pt':
                self.model.load_state_dict(torch.load(args.resume, self.device))
                
                
        ###initial bayesian loss, post probs

        self.post_prob = Post_Prob(args.sigma,
                                   args.crop_size,
                                   args.downsample_ratio,
                                   args.background_ratio,
                                   args.use_background,
                                   self.device)
        self.criterion = Bay_Loss(args.use_background, self.device)#####bayes loss
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0
        

    def train(self):
        """training process"""
        args = self.args
        features1 = []
        features2 = []
        for epoch in range(self.start_epoch, args.max_epoch):
        
            
        
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            self.train_eopch(self.epoch)
            #print(f1.shape)
            self.scheduler.step()
            
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()


    def train_eopch(self, epoch):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        
        ##### Set model to training mode
        
        self.model.train()  

        # Iterate over data.
        for step, (inputs, points, prior_prob, st_sizes, grid_c, gridnum_sam_c, gd_count) in enumerate(self.dataloaders['train']): 
            #print(inputs.shape, sr_gt.shape)#3 256 256
            #print(targets.shape, st_sizes)
            
            ###set data to device
            
            inputs = inputs.to(self.device) ###img
            points = [p.to(self.device) for p in points] ###points
            prior_prob = [t.to(self.device) for t in prior_prob] ###prior_prob
            st_sizes = st_sizes.to(self.device)
            
            grid_c = grid_c.to(self.device)
            
            
            gridnum_sam_c = [tt.to(self.device) for tt in gridnum_sam_c]
            

            
            
            ###iteration
            with torch.set_grad_enabled(True):
            
                ###run the model
                use_sr = False

                outputs, out_sigma = self.model(inputs, grid_c, 'train', epoch)   
  
                  
                ###bayesian loss
                prob_list = self.post_prob(points, st_sizes, gridnum_sam_c)


                loss1 = self.criterion(prob_list, prior_prob, outputs, epoch) 
                
                loss_KL = self.model.kl_div
                
                
                outputs = outputs/10
                
                
                
                if False:
                
                  loss = 1.0*loss1 + 0.01*loss_KL
                
                  if epoch<500:
                      
                      
                      if epoch%5 == 1:
                          self.optimizer1.zero_grad()
                          self.optimizer2.zero_grad()
                          loss.backward()
                          self.optimizer2.step()
                      else:
                          self.optimizer1.zero_grad()
                          self.optimizer2.zero_grad()
                          loss.backward()
                          self.optimizer1.step()
                  else:
                      self.optimizer1.zero_grad()
                      self.optimizer2.zero_grad()
                      loss.backward()
                      self.optimizer1.step()
                      self.optimizer2.step()
                else:
                
                  loss = 1.0*loss1#+ 0.1*loss_KL
                
                  self.optimizer1.zero_grad()
                  self.optimizer2.zero_grad()
                    
                  loss.backward()
                  self.optimizer1.step()
                  self.optimizer2.step()
                  

                
                #for p in self.model.resnet_backbone.frontend2.parameters():### loss1: 10e-2, loss2: >10e2
                #    if p is not None:
                #        print(p.grad)
                
                ####update the metrics
                N = inputs.size(0)
                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                #print(pre_count)
                #print(pre_count.shape, gd_count.shape)
                
                res = pre_count - gd_count

                #print(np.sum(f1),np.sum(f2))
                

                #print(pre_count, gd_count)
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)


        
        
        logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                             time.time()-epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            #'optimizer_state_dict': self.optimizer.state_dict(),
            'optimizer_state_dict1': self.optimizer1.state_dict(),
            'optimizer_state_dict2': self.optimizer2.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models
        
        

    def val_epoch(self):
        
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        
        sr_results = []
        c_results = []
        
        # Iterate over data.
        for inputs, count, name, cor_C in self.dataloaders['val']:
            #print('val input',inputs.shape)
            inputs = inputs.to(self.device)
            cor_C = cor_C.to(self.device)
            # inputs are images with different sizes
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                #print(cor_C)

                outputs, out_sigma = self.model(inputs, cor_C, 'test',self.epoch)
                    

                outputs = outputs/10


                res = count[0].item() - torch.sum(outputs).item()
                #print(torch.sum(outputs).item(),count[0].item())

                epoch_res.append(res)

                c_results.append(outputs.data.cpu().numpy())
                
        #print(torch.sqrt(torch.sum(0.01*torch.pow(out_sigma,2))).data.cpu().numpy())

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        #np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg()
        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, time.time()-epoch_start))
                     

        
        model_state_dic = self.model.state_dict()
        #print(model_state_dic)
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
        #if mae < self.best_mae:
        #if True:
            self.best_mse = mse
            self.best_mae = mae
            logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                 self.best_mae,
                                                                                 self.epoch))
            #torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))
            torch.save(self.model, os.path.join(self.save_dir, 'best_model.pt'))
            
            sr_results = np.array(sr_results)
            c_results = np.array(c_results)
            np.save('srr_image.npy', sr_results)
            np.save('cc_image.npy', c_results)
        logging.info("best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                 self.best_mae,
                                                                                 self.epoch))
        print(self.best_mae, self.best_mse)
        if mae < 7.5:
            #torch.save(model_state_dic, os.path.join(self.save_dir, 'recent_model.pth'))
            torch.save(self.model, os.path.join(self.save_dir, 'recent_model.pt'))
        
        if mae < 7.2:
            #torch.save(model_state_dic, os.path.join(self.save_dir, 'recent_model.pth'))
            #torch.save(self.model, os.path.join(self.save_dir, str(self.epoch)+'fine_model.pt'))
            torch.save(self.model, os.path.join(self.save_dir, 'fine_model.pt'))
            
        c_results = np.array(c_results)    
        if self.epoch %10 == 0:
            fig = plt.figure(figsize=(10, 7))
            rows = 1
            columns = 4
            for i in range(4):
                c_example = np.reshape(c_results[i], [32,32])
                fig.add_subplot(rows, columns, i+1)    
                plt.imshow(c_example)
            plt.savefig(os.path.join(self.save_dir, str(self.epoch)+'_test_fig.png'))
            plt.close()
        
        
        
        
        



