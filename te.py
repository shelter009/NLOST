from util.LFEDataset import LFEDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import scipy.io as sio
import models.nlost
folder_path = ["D:\\NLOS\\bike\\bike"]
shineness = [0]

val_data = LFEDataset(root=folder_path,  # dataset root directory
                      shineness=shineness,
                      for_train=False,
                      ds=1,  # temporal down-sampling factor
                      clip=512,  # time range of histograms
                      size=256,  # measurement size (unit: px)
                      scale=1,  # scaling factor (float or float tuple)
                      background=[0.05, 2],  # background noise rate (float or float tuple)
                      target_size=256,  # target image size (unit: px)
                      target_noise=0.01,  # standard deviation of target image noise
                      color='gray')  # color channel(s) of target image

#print(len(val_data))

# 将 num_workers 参数设置为 0 或者删除，默认值即为 0
#val_loader = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=True)
#print(len(val_loader))


#x = cv2.imread("confocal-0-0.2649-0.0000.hdr", cv2.IMREAD_UNCHANGED)
#x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
#x = x.reshape(x.shape[1], x.shape[1], 1)  # 256 256 1 1
#x = cv2.resize(x, (128,128))
#x = x.reshape(1, x.shape[1], x.shape[1])  # 128 128 1 1
#x = x / np.max(x)
#print(x.shape)
#path = "D:\\NLOS\\Unseen_spad\\0\\1a0bc9ab92c915167ae33d942430658c\\shine_0.0000-rot_-0.7742_-100.1241_-4.5676-shift_0.2157_0.1523_-0.3004.mat"
#x = sio.loadmat(
#                    path, verify_compressed_data_integrity=False
#                )['data']
#print(x.shape)
model = models.nlost.NLOST()
x = torch.rand((2,1,512,256,256))
a  = model(x)
print(a.shape)