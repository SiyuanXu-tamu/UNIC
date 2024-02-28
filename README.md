###  Official Implement paper "Uncertainty-aware Continuous Implicit Neural Representations for Remote Sensing Object Counting"



## Code

### Install dependencies

torch >= 1.0 torchvision opencv numpy scipy, all the dependencies can be easily installed by pip or conda

This code was tested with python 3.6  

###  Train and Test

1、 Dowload Dataset RSOC [Link](https://pan.baidu.com/s/19hL7O1sP_u2r9LNRsFSjdA)  code：nwcx

2、 Pre-Process Data (resize image and split train/validation)

```
python preprocess_dataset.py --origin_dir <directory of original data> --data_dir <directory of processed data>
```
3、 generate ground truth for our loss function

```
python preprocess_gt.py
```
3、 Train model (validate on single NVidia V100)

```
python train.py --data_dir <directory of processed data> --save_dir <directory of log and model>
```

4、 Test Model
```
python val_test.py
```

The result is slightly influenced by the random seed, but fixing the random seed (have to set cuda_benchmark to False) will make training time extrodinary long, so sometimes you can get a slightly worse result than the reported result, but most of time you can get a better result than the reported one. If you find this code is useful, please give us a star and cite our paper, have fun.



### Pretrain Weight
#### RSOC-building


Google Drive [Link](https://drive.google.com/file/d/1d9VN-_o5-IVw_-yUpWXrkQhvfS3CZpQO/view?usp=share_link)




