###  Official Implement paper "Uncertainty-aware Continuous Implicit Neural Representations for Remote Sensing Object Counting"



## Code

### Install dependencies

torch >= 1.0 torchvision opencv numpy scipy, all the dependencies can be easily installed by pip or conda

This code was tested with python 3.6  

###  Train and Test

1、 Dowload Dataset RSOC [Link](https://pan.baidu.com/s/19hL7O1sP_u2r9LNRsFSjdA)  code：nwcx

2、 Pre-Process Data (resize image and split train/validation)

```
python preprocess_dataset.py
```
3、 generate ground truth for our loss function

```
python preprocess_gt.py, python make_dataset_RSOC.py
```
3、 Train model (validate on single NVidia V100)

```
python train.py
```


The result is slightly influenced by the random seed, but fixing the random seed (have to set cuda_benchmark to False) will make training time extrodinary long, so sometimes you can get a slightly worse result than the reported result, but most of time you can get a better result than the reported one. If you find this code is useful, please give us a star and cite our paper, have fun.
Also, this code also gives 2 kinds of UNIC that you can try, they are in models/aspd_spatial_uq1.py and models/aspd_spatial_uq2.py.





