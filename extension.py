import cv2
from PIL import Image
import os
import numpy as np
# def flip_images(image_folder):
#     """
#     对文件夹下的图片进行翻转操作（水平、垂直、水平垂直翻转）
#     """
#     for filename in os.listdir(image_folder):
#         if filename.endswith(('.jpg', '.png', '.jpeg')):
#             img_path = os.path.join(image_folder, filename)
#             img = cv2.imread(img_path)
#             # 水平翻转
#             horizontal_flip = cv2.flip(img, 1)
#             # 垂直翻转
#             vertical_flip = cv2.flip(img, 0)
#             # 水平垂直翻转
#             both_flip = cv2.flip(img, -1)
#
#             # 保存翻转后的图片，可根据实际需求自定义保存路径和文件名格式
#             cv2.imwrite(os.path.join(image_folder, "horizontal_" + filename), horizontal_flip)
#             cv2.imwrite(os.path.join(image_folder, "vertical_" + filename), vertical_flip)
#             cv2.imwrite(os.path.join(image_folder, "both_" + filename), both_flip)
def resize_images(image_folder, new_size=(200, 200)):
    """
    对文件夹下的图片进行缩放操作
    :param image_folder: 图片所在文件夹路径
    :param new_size: 缩放后的尺寸，默认为(200, 200)
    """
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path)
            resized_img = img.resize(new_size)
            resized_img.save(os.path.join(image_folder, "resized_" + filename))
def rotate_images(image_folder, angle=45):
    """
    对文件夹下的图片进行旋转操作
    :param image_folder: 图片所在文件夹路径
    :param angle: 旋转角度，默认为45度
    """
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_img = cv2.warpAffine(img, rotation_matrix, (w, h))
            cv2.imwrite(os.path.join(image_folder, "rotated_" + filename), rotated_img)
def adjust_brightness_contrast(image_folder, alpha=1.5, beta=30):
    """
    对文件夹下的图片进行亮度及对比度调整操作
    :param image_folder: 图片所在文件夹路径
    :param alpha: 对比度控制因子，大于1增加对比度，默认为1.5
    :param beta: 亮度调整值，正值增加亮度，默认为30
    """
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)
            adjusted_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            cv2.imwrite(os.path.join(image_folder, "adjusted_" + filename), adjusted_img)
def crop_images(image_folder, crop_size=(100, 100)):
    """
    对文件夹下的图片进行剪裁操作
    :param image_folder: 图片所在文件夹路径
    :param crop_size: 剪裁尺寸，默认为(100, 100)
    """
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path)
            width, height = img.size
            left = (width - crop_size[0]) // 2
            top = (height - crop_size[1]) // 2
            right = left + crop_size[0]
            bottom = top + crop_size[1]
            cropped_img = img.crop((left, top, right, bottom))
            cropped_img.save(os.path.join(image_folder, "cropped_" + filename))
if __name__ == "__main__":
    image_folder = "C:/Users/25507/Desktop/train-sign/10"  # 替换为实际的图片文件夹路径
    #flie_images(image_folder)
    resize_images(image_folder)
    rotate_images(image_folder)
    adjust_brightness_contrast(image_folder)
    crop_images(image_folder)