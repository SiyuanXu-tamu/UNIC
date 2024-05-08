from utils.regression_trainer_unic import RegTrainer

import argparse
import os
import torch

import random
import numpy as np


args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--data-dir', default='../dataset/new_building/building_256/',
                        help='training data directory')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save models.')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='the weight decay')
    parser.add_argument('--resume', default='',#'pretrained_model/1975_ckpt.tar'
                        help='the path of resume training model')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=1500,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=2, 
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=1,
                        help='the epoch start to val')

    parser.add_argument('--batch-size', type=int, default=16,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')

    parser.add_argument('--is-gray', type=bool, default=False,
                        help='whether the input image is gray')
    parser.add_argument('--crop-size', type=int, default=256,
                        help='the crop size of the train image')
    parser.add_argument('--downsample-ratio', type=int, default=8,
                        help='downsample ratio')

    parser.add_argument('--use-background', type=bool, default=True,
                        help='whether to use background modelling')
    parser.add_argument('--sigma', type=float, default=4.0,
                        help='sigma for likelihood')
    parser.add_argument('--background-ratio', type=float, default=1.0,
                        help='background ratio')
                        
    parser.add_argument('--use_sr', type=bool, default=False,
                        help='use_sr')
    parser.add_argument('--seed', type=int, default=64,
                        help='seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    #torch.backends.cudnn.benchmark = True
    
    random.seed(args.seed)     # python random generator
    np.random.seed(args.seed)  # numpy random generator
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()
