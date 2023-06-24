import cv2
import numpy as np
import numpy.fft as fft

from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
from matplotlib import  rcParams


def rgb_dct2(rgb_image: np.ndarray):
    '''
    对 rgb_image 作 dct 变换，返回频域图像
    '''
    return np.transpose([cv2.dct(np.float32(rgb_image[:, :, i])) for i in range(3)], axes=(1, 2, 0))


def rgb_idct2(rgb_image_dct: np.ndarray):
    '''
    对 rgb_image_dct 作逆 dct 变换，返回空域图像
    '''
    Img_idct2 = np.transpose([cv2.idct(rgb_image_dct[:, :, i]) for i in range(3)],
                             axes=(1, 2, 0))
    Img_idct2_qua = Img_idct2.copy()
    Img_idct2_qua[Img_idct2_qua > 255] = 255
    Img_idct2_qua[Img_idct2_qua <   0] = 0
    Img_idct2_qua = np.uint8(Img_idct2_qua)
    return Img_idct2_qua


def get_feature(Image: np.ndarray, blocks: tuple):
    '''
    对图像 Image 按 blocks 进行分块，返回每个块的最大值
    '''
    block_shape = (Image.shape[0]//blocks[0], Image.shape[1]//blocks[1])
    feature = np.zeros(
        shape=(blocks[0], blocks[1], 3))
    for i in range(feature.shape[0]):
        for j in range(feature.shape[1]):
            feature[i, j, :] = np.max(
                Image[block_shape[0]*i : block_shape[0]*(i+1),
                      block_shape[1]*j : block_shape[1]*(j+1),
                      :],
                axis=(0, 1))
    feature = np.uint8((feature - np.min(feature))*255 //
                       (np.max(feature) - np.min(feature)))
    return feature


def differ(img1_fea: np.ndarray, img2_fea: np.ndarray, return_pos=True):
    sum = 0
    position = []
    for i in range(img1_fea.shape[0]):
        for j in range(img1_fea.shape[1]):
            if (np.max(np.abs(img1_fea[i, j, :] - img2_fea[i, j, :])) > 100):
                sum += 1
                if (return_pos == True):
                    position.append((i, j))
    if (return_pos == True):
        return position, sum / (img1_fea.shape[0] * img1_fea.shape[1])
    else:
        return sum / (img1_fea.shape[0] * img1_fea.shape[1])


def extract(Image: np.ndarray, blocks: tuple):
    '''
    对图像 Image 按 blocks 进行分块，提取每个块的特征水印
    '''
    block_size = (Image.shape[0]//blocks[0], Image.shape[1]//blocks[1])
    wm_ext = np.zeros(shape=blocks)
    for i in range(blocks[0]):
        for j in range(blocks[1]):
            block = Image[i * block_size[0] : (i+1) * block_size[0],
                          j * block_size[1] : (j+1) * block_size[1], :]
            block_dct = rgb_dct2(block)
            wm_ext[i, j, :] = block_dct[-1, -1, :]
    return wm_ext


def mark_diff(Image: np.ndarray, position: list, blocks: tuple):
    '''
    在图像 Image 上用方框标记 position 对应的的块
    '''
    block_size = [Image.shape[0]//blocks[0], Image.shape[1]//blocks[1]]
    Image_marked = Image.copy()
    for pos in position:
        start_point = (pos[1] * block_size[1] + 2, pos[0] * block_size[0] + 2)
        end_point = (pos[1] * block_size[1] + block_size[1] - 2, pos[0] * block_size[0] + block_size[0] - 2)
        cv2.rectangle(Image_marked, start_point, end_point,
                      (0, 0, 255), thickness=2, lineType=cv2.LINE_8)
    return Image_marked
