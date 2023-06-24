import utility
import wama_fft
import wama_dct

import gradio as gr

import cv2
import numpy as np
import numpy.fft as fft

from PIL import Image, ImageDraw, ImageFont, ImageColor

import matplotlib.pyplot as plt
from matplotlib import  rcParams

rcParams['font.family'] = 'monospace'
rcParams['font.monospace'] = ['Ubuntu Mono', 'JetBrains Mono', 'Consolas', 'monospace']


WATER_MARK_ALPHA = 0.04
TAMPER_LOCATE_CROP = 0.05

def crop_handler(image):
    return image, "Crop Done!"

def sketch_handler(image, color):
    img, mask = image["image"], image["mask"]
    mask = mask[..., 0] > 128
    # 纯色图片
    color = ImageColor.getcolor(color, "RGB")
    img_sketch = np.full_like(img, color)

    img[mask] = img_sketch[mask]
    return img, "Sketch Done!"

def rotate_handler(image, angle):
    img_rot = cv2.warpAffine(image, cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1), (image.shape[1], image.shape[0]))
    return img_rot, "Rotate Done!"

def blur_handler(image, blur_size):
    img, mask = image["image"], image["mask"]
    mask = mask[..., 0] > 128

    img_blur = cv2.GaussianBlur(img, (blur_size, blur_size), 0)

    img[mask] = img_blur[mask]

    return img, "Blur Done!"

def config_handler(wm_alpha, tl_crop):
    WATER_MARK_ALPHA = wm_alpha
    TAMPER_LOCATE_CROP = tl_crop
    return "WATER_MARK_ALPHA = {}\nTAMPER_LOCATE_CROP = {}".format(WATER_MARK_ALPHA, TAMPER_LOCATE_CROP)

def wm_fft(image_ori_input: np.ndarray, watermark_text: str):

    image_ori = np.uint8(image_ori_input.copy())

    image_fft = wama_fft.rgb_fft2(image_ori)

    # 裁剪 image_fft_shift 周围 TAMPER_LOCATE_CROP 范围，并在裁剪范围内插入水印
    image_wm_fft = image_fft.copy()
    # 裁剪像素 = 原图窄边 * TAMPER_LOCATE_CROP
    crop_width = int(TAMPER_LOCATE_CROP * min(image_wm_fft.shape[0], image_wm_fft.shape[1]))
    # 裁剪 image_wm4_fft，将四边 crop_width 内全部清零
    image_wm_fft[:crop_width , :] = 0
    image_wm_fft[-crop_width:, :] = 0
    image_wm_fft[:, :crop_width ] = 0
    image_wm_fft[:, -crop_width:] = 0

    # 在此基础上加上水印
    water_mark = utility.make_water_mark(image_ori, utility.load_water_mark('text', watermark_text, 35), (crop_width, crop_width))
    image_wm_fft = image_wm_fft * (1 + WATER_MARK_ALPHA * (water_mark[:, :, :1]))

    image_wm = wama_fft.rgb_ifft2(image_wm_fft)
    return utility.center_log_spectrum(image_fft), water_mark, utility.center_log_spectrum(image_wm_fft), image_wm

def iwm_fft(image_wm_input: np.ndarray):
    img_wm_fft = wama_fft.rgb_fft2(image_wm_input)

    # 提取 CROP 范围内的频谱
    crop_width = int(TAMPER_LOCATE_CROP * min(img_wm_fft.shape[0], img_wm_fft.shape[1]))
    img_wm_fft_crop = img_wm_fft.copy()
    img_wm_fft_crop[crop_width:-crop_width, crop_width:-crop_width] = 0

    # 逆变换
    img_tamper_locate = wama_fft.rgb_ifft2(img_wm_fft_crop) * 64
    return utility.center_log_spectrum(img_wm_fft), utility.center_log_spectrum(img_wm_fft_crop), img_tamper_locate

def wm_dct(image_ori_input: np.ndarray, watermark_text: str):
    pass

def iwm_dct(image_wm_input: np.ndarray):
    pass

with gr.Blocks() as app:
    gr.Markdown("# Img Watermark")

    with gr.Tab("DFT"):

        with gr.Accordion("DFT 全局参数", open = False):

            gr.Markdown("*以下三个模块的运行请在同一全局参数下进行*")
            wm_alpha = gr.Slider(minimum=0, maximum=0.15, step=0.005, label="水印强度", value=0.04)
            tl_crop = gr.Slider(minimum=0, maximum=0.2, step=0.005, label="篡改定位裁剪阈值", value=0.05)
            config_msg = gr.Textbox(placeholder="WATER_MARK_ALPHA = {}\nTAMPER_LOCATE_CROP = {}".format(WATER_MARK_ALPHA, TAMPER_LOCATE_CROP), label="Config Message", lines=2, interactive=False)
            config_button = gr.Button("Submmit")

        with gr.Accordion("水印添加模块", open = True):

            with gr.Row():

                with gr.Column():
                    gr.Markdown("## 输入")

                    gr.Markdown("### 图像与水印输入")
                    image_ori_input = gr.Image(label = "上传原图")
                    watermark_text = gr.Textbox(placeholder="输入水印内容", label="Watermark Content", lines=1)
                    wm_fft_button = gr.Button("Make Watermark")

                with gr.Column():
                    gr.Markdown("## 输出")

                    gr.Markdown("### 预处理")
                    with gr.Row():
                        image_fft_show = gr.Image(label = "原图频域 (dft)")
                        water_mark_show = gr.Image(label = "水印")
                    gr.Markdown("### 频域水印添加")
                    with gr.Row():
                        image_fft_wm_show = gr.Image(label = "原图频域 + 水印 (dft)")
                        image_wm_show = gr.Image(label = "原图 + 水印 空域 (idft)")

        with gr.Accordion("图片篡改小工具",open = False):

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 裁剪 Crop")
                    image_process_crop_input = gr.Image(label = "上传图像", tool="select")
                    fft_crop_button = gr.Button("Crop")

                with gr.Column():
                    gr.Markdown("### 绘制 Sketch")
                    image_process_sketch_input = gr.Image(label = "上传图像", tool="sketch")
                    skecth_color = gr.ColorPicker(label="画笔颜色", value="black")
                    fft_sketch_button = gr.Button("Sketch")

                with gr.Column():
                    gr.Markdown("### 旋转 Rotate")
                    image_process_rotate_input = gr.Image(label = "上传图像")
                    rotate_angle = gr.Slider(minimum=-180, maximum=180, step=1, label="旋转角度", value=20)
                    fft_rotate_button = gr.Button("Rotate")

                with gr.Column():
                    gr.Markdown("### 模糊 Blur")
                    image_process_blur_input = gr.Image(label = "上传图像", tool="sketch", source="upload")
                    blur_radius = gr.Slider(minimum=1, maximum=21, step=2, label="模糊半径", value=5)
                    fft_blur_button = gr.Button("Blur")

            gr.Markdown("## 输出")
            with gr.Row():
                image_process_msg = gr.Textbox(placeholder="Output Message", label="输出信息", lines=1, interactive=False)
                image_process_show = gr.Image(label = "处理后图像")

        with gr.Accordion("水印提取与篡改定位模块",open = False):

            with gr.Row():

                with gr.Column():
                    gr.Markdown("## 输入")

                    gr.Markdown("### 水印图像输入")
                    image_wm_input = gr.Image(label = "上传水印后图像")
                    iwm_fft_button = gr.Button("Show Watermark & Tamper Locate")

                with gr.Column():
                    gr.Markdown("## 输出")

                    gr.Markdown("### 水印显示与篡改信息提取")
                    with gr.Row():
                        image_wm_fft_show = gr.Image(label = "水印图像频域 (dft)")
                        water_wm_window_show = gr.Image(label = "篡改定位信息窗")
                    gr.Markdown("### 篡改定位显示")
                    image_tamper_locate_show = gr.Image(label = "原图频域 + 水印 (dft)")

        config_button.click(config_handler, inputs=[wm_alpha, tl_crop], outputs=[config_msg])

        wm_fft_button.click(wm_fft, inputs=[image_ori_input, watermark_text], outputs=[image_fft_show, water_mark_show, image_fft_wm_show, image_wm_show])
        iwm_fft_button.click(iwm_fft, inputs=[image_wm_input], outputs=[image_wm_fft_show, water_wm_window_show, image_tamper_locate_show])

        fft_crop_button.click(crop_handler, inputs=[image_process_crop_input], outputs=[image_process_show, image_process_msg])
        fft_sketch_button.click(sketch_handler, inputs=[image_process_sketch_input, skecth_color], outputs=[image_process_show, image_process_msg])
        fft_rotate_button.click(rotate_handler, inputs=[image_process_rotate_input, rotate_angle], outputs=[image_process_show, image_process_msg])
        fft_blur_button.click(blur_handler, inputs=[image_process_blur_input, blur_radius], outputs=[image_process_show, image_process_msg])

    with gr.Tab("DCT"):
        with gr.Accordion("水印添加模块", open = False):

            with gr.Row():

                with gr.Column():
                    gr.Markdown("## 输入")

                    gr.Markdown("### 图像与水印输入")
                    image_ori_input = gr.Image(label = "上传原图")
                    watermark_text = gr.Textbox(placeholder="输入水印内容", label="Watermark Content", lines=1)

                    wm_dct_button = gr.Button("Make Watermark")

                with gr.Column():
                    gr.Markdown("## 输出")

                    gr.Markdown("### 预处理")
                    with gr.Row():
                        image_dct_show = gr.Image(label = "原图频域 (dft)")
                        water_mark_show = gr.Image(label = "水印")
                    gr.Markdown("### 频域水印添加")
                    with gr.Row():
                        image_dct_wm_show = gr.Image(label = "原图频域 + 水印 (dft)")
                        image_wm_show = gr.Image(label = "原图 + 水印 空域 (idft)")

        with gr.Accordion("图片篡改小工具",open = False):

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 裁剪 Crop")
                    image_process_crop_input = gr.Image(label = "上传图像", tool="select")
                    dct_crop_button = gr.Button("Crop")

                with gr.Column():
                    gr.Markdown("### 绘制 Sketch")
                    image_process_sketch_input = gr.Image(label = "上传图像", tool="sketch")
                    skecth_color = gr.ColorPicker(label="画笔颜色", value="black")
                    dct_sketch_button = gr.Button("Sketch")

                with gr.Column():
                    gr.Markdown("### 旋转 Rotate")
                    image_process_rotate_input = gr.Image(label = "上传图像")
                    rotate_angle = gr.Slider(minimum=-180, maximum=180, step=1, label="旋转角度", value=20)
                    dct_rotate_button = gr.Button("Rotate")

                with gr.Column():
                    gr.Markdown("### 模糊 Blur")
                    image_process_blur_input = gr.Image(label = "上传图像", tool="sketch", source="upload")
                    blur_radius = gr.Slider(minimum=1, maximum=21, step=2, label="模糊半径", value=5)
                    dct_blur_button = gr.Button("Blur")

            gr.Markdown("## 输出")
            with gr.Row():
                image_process_msg = gr.Textbox(placeholder="Output Message", label="输出信息", lines=1, interactive=False)
                image_process_show = gr.Image(label = "处理后图像")

        with gr.Accordion("水印提取与篡改定位模块",open = False):

            with gr.Row():

                with gr.Column():
                    gr.Markdown("## 输入")

                    gr.Markdown("### 水印图像输入")
                    image_wm_input = gr.Image(label = "上传水印后图像")
                    iwm_dct_button = gr.Button("Show Watermark & Tamper Locate")

                with gr.Column():
                    gr.Markdown("## 输出")

                    gr.Markdown("### 水印显示与篡改信息提取")
                    with gr.Row():
                        image_wm_dct_show = gr.Image(label = "水印图像频域 (dft)")
                        water_wm_window_show = gr.Image(label = "篡改定位信息窗")
                    gr.Markdown("### 篡改定位显示")
                    image_tamper_locate_show = gr.Image(label = "原图频域 + 水印 (dft)")

        wm_dct_button.click(wm_dct, inputs=[image_ori_input, watermark_text], outputs=[image_dct_show, water_mark_show, image_dct_wm_show, image_wm_show])
        iwm_dct_button.click(iwm_dct, inputs=[image_wm_input], outputs=[image_wm_dct_show, water_wm_window_show, image_tamper_locate_show])

        dct_crop_button.click(crop_handler, inputs=[image_process_crop_input], outputs=[image_process_show, image_process_msg])
        dct_sketch_button.click(sketch_handler, inputs=[image_process_sketch_input, skecth_color], outputs=[image_process_show, image_process_msg])
        dct_rotate_button.click(rotate_handler, inputs=[image_process_rotate_input, rotate_angle], outputs=[image_process_show, image_process_msg])
        dct_blur_button.click(blur_handler, inputs=[image_process_blur_input, blur_radius], outputs=[image_process_show, image_process_msg])

app.launch(share=False)