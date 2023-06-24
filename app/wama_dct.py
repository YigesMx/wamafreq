import cv2
import numpy as np
import numpy.fft as fft

from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
from matplotlib import  rcParams

def rgb_dct2(rgbImage:np.ndarray):
    '''
    对rgbImage作dct变换，返回数据类型为float32的频域图像
    '''
    return np.transpose([cv2.dct(np.float32(rgbImage[:,:,i])) for i in range(3)],axes=(1,2,0))
def rgb_idct2(rgbImage_dct:np.ndarray):
    '''
    对rgbImage_idct作逆dct变换，返回数据类型为uint8的空域图像
    '''
    Img_idct2 = np.transpose([cv2.idct(rgbImage_dct[:,:,i]) for i in range(3)],axes=(1,2,0))
    Img_idct2_qua = Img_idct2.copy()
    for i in range(Img_idct2_qua.shape[0]):
        for j in range(Img_idct2_qua.shape[1]):
            for k in range(Img_idct2_qua.shape[2]):
                if(Img_idct2_qua[i,j,k]>255):
                    Img_idct2_qua[i,j,k]=255
                elif(Img_idct2_qua[i,j,k]<0):
                    Img_idct2_qua[i,j,k]=0
                else:
                    Img_idct2_qua[i,j,k]=round(Img_idct2_qua[i,j,k])
    Img_idct2_qua = np.uint8(Img_idct2_qua)
    return Img_idct2_qua

def get_feature(Image:np.ndarray,block_size:tuple):
    feature = np.zeros(shape=(Image.shape[0]//block_size[0],Image.shape[1]//block_size[1],3))
    for i in range(feature.shape[0]):
        for j in range(feature.shape[1]):
            feature[i,j,:] = np.max(  Image[block_size[0]*i : block_size[0]*(i+1),block_size[1]*j : block_size[1]*(j+1),:],axis=(0,1))
    feature = np.uint8((feature - np.min(feature))*255 // (np.max(feature) - np.min(feature)))
    return feature

def differ(img1_fea:np.ndarray,img2_fea:np.ndarray,return_pos=True):
    sum = 0
    position = []
    for i in range(img1_fea.shape[0]):
        for j in range(img1_fea.shape[1]):
            if(np.max(np.abs(img1_fea[i,j,:]-img2_fea[i,j,:])) > 100):
                sum = sum+1
                if(return_pos == True):
                    position.append((i,j))
    if(return_pos == True):
        return position,sum/(img1_fea.shape[0]*img1_fea.shape[1])
    else:
        return sum/(img1_fea.shape[0]*img1_fea.shape[1])


def extract(Image:np.ndarray,wm_size:tuple):
    wm_ext = np.zeros(shape=wm_size)
    Image_dct = rgb_dct2(Image)
    for i in range(wm_size[0]):
        for j in range(wm_size[1]):
            block = Image[i*block_size[0]:(i+1)*block_size[0],j*block_size[1]:(j+1)*block_size[1],:]
            block_dct = rgb_dct2(block)
            wm_ext[i,j,:] = block_dct[-1,-1,:] 
    return wm_ext

def mark_diff(Image,position,block_shape):
    Image_marked = Image.copy()
    for pos in position:
        start_point = (pos[1]*block_shape[0],pos[0]*block_shape[1])
        end_point = (pos[1] * block_shape[0] + block_shape[0] - 4, pos[0] * block_shape[1] + block_shape[1] - 4)
        cv2.rectangle(Image_marked, start_point, end_point, (0, 0, 255), thickness=2, lineType=cv2.LINE_8)
    return Image_marked
