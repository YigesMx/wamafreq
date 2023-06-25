import cv2
import numpy as np
import numpy.fft as fft

from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
from matplotlib import  rcParams

rcParams['font.family'] = 'monospace'
rcParams['font.monospace'] = ['Ubuntu Mono', 'JetBrains Mono', 'Consolas', 'monospace']

def text_to_img(text, img_height):
    """ 文字转图片, 字体大小填充图片高度 """
    font = ImageFont.truetype('consola', img_height)
    img_width = int(font.getlength(text))
    img = Image.new('RGB', (img_width, img_height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font, fill=(255, 255, 255))
    return np.array(img)


def load_water_mark(type: str, name: str, height: int = None):
    """ 载入适合频域的水印
    如果是 type 为 text 则生成内容为 name 的图片作为水印
    如果是 type 为 image 则载入目录下名称为 name 的图片作为水印, 将图片等比缩放至高度为 height """
    if type == 'text':
        wm = text_to_img(name, height)
    elif type == 'image':
        wm = cv2.imread(name)
        if height is not None:
            wm = cv2.resize(
                wm, (int(wm.shape[1] * height / wm.shape[0]), height))
    else:
        raise Exception('type must be text or image')
    return wm


def make_water_mark(target: np.ndarray, water_mark: np.ndarray, offset: tuple = (0, 0)):
    """ 制作水印
    创建 target 大小的图片 """
    wm = np.zeros(target.shape, dtype=np.uint8)
    # 制作
    wm[-water_mark.shape[0]-offset[0]:-offset[0], - water_mark.shape[1]-offset[1]:-offset[1]] = water_mark
    wm = wm[::-1, ::-1]
    wm[-water_mark.shape[0]-offset[0]:-offset[0], - water_mark.shape[1]-offset[1]:-offset[1]] = water_mark
    wm = wm[::-1, ::-1]
    return np.uint8(wm)


def show_images(images: list, titles: list, n: int, m: int, font_scale: float = 1, dpi: int = 800):
    """ 图像显示辅助函数
    并排显示 n * m 张图像, n 行 m 列
    在每张图左侧和顶侧显示像素坐标, 在每张图顶侧显示标题 """
    plt.figure(dpi=dpi)
    # 调整文字大小
    rcParams['font.size'] = 12 * font_scale
    for i in range(n):
        for j in range(m):
            plt.subplot(n, m, i * n + j + 1)
            plt.imshow(images[i * n + j], cmap='gray')
            plt.title(titles[i * n + j])
    plt.show()

def center_log_spectrum(input: np.ndarray):
    """ 对数化、归一化、* 255 """
    spectrum = np.log(np.abs(input) + 1)
    spectrum = (spectrum - np.min(spectrum)) * 255 // (np.max(spectrum) - np.min(spectrum))
    return np.uint8(spectrum)