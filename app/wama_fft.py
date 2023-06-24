import cv2
import numpy as np
import numpy.fft as fft

from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
from matplotlib import  rcParams

rcParams['font.family'] = 'monospace'
rcParams['font.monospace'] = ['Ubuntu Mono', 'JetBrains Mono', 'Consolas', 'monospace']


def rgb_fft2(rgbImage):
    """ 对 rgb 图像分通道做 fft2, 并将高频分量移至中心 """
    if (rgbImage.__class__ == np.ndarray):
        return fft.fftshift(fft.fft2(rgbImage, axes=(0, 1)), axes=(0, 1))
    elif (rgbImage.__class__ == list):
        return [fft.fftshift(fft.fft2(i, axes=(0, 1)), axes=(0, 1)) for i in rgbImage]
    else:
        return None


def rgb_ifft2(rgbImage_fft_shift):
    if (rgbImage_fft_shift.__class__ == np.ndarray):
        ifft = fft.ifft2(fft.ifftshift(rgbImage_fft_shift, axes=(0, 1)), axes=(0, 1))
        ifft[ifft > 255] = 255
        ifft[ifft < 0] = 0
        return np.uint8(ifft.real)
    elif (rgbImage_fft_shift.__class__ == list):
        ifft = [fft.ifft2(fft.ifftshift(i, axes=(0, 1)), axes=(0, 1)) for i in rgbImage_fft_shift]
        ifft = [np.uint8(i.real) for i in ifft]
        ifft[ifft > 255] = 255
        ifft[ifft < 0] = 0
        return ifft
    else:
        return None