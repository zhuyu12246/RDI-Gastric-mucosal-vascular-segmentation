# 蓝色粗血管分割
# Blue thick blood vessel segmentation
# images/result_laster.png
import cv2
import numpy as np


# 读取图像
img = cv2.imread('images/result_laster.png')

# 转为灰度图（血管在红色色通道最明显）
blue_channel = img[:,:,2]

# 使用高斯滤波去噪
blur = cv2.GaussianBlur(blue_channel, (5,5), 0)

# 局部对比度增强（CLAHE）
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(blur)

_, mask = cv2.threshold(enhanced, 100, 255, cv2.THRESH_BINARY_INV)
# mask[blue_channel < 60] = 0

# 形态学
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

# 血管增强/连续
mask_enhanced = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations=2)

# 去噪
mask_clean = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel,iterations=2)

# 创建彩色血管图像（红色）
mask_colored = np.zeros_like(img)
mask_colored[mask_clean == 255] = [0, 0, 255]  # BGR红色
# 半透明叠加
alpha = 0.3  # 血管透明度，0~1
overlay = cv2.addWeighted(img, 1.0, mask_colored, alpha, 0)

# cv2.imshow('blue_channel', blue_channel)
cv2.imshow('blur', blur)
cv2.imshow('enhanced', enhanced)
cv2.imshow('mask', mask)
cv2.imshow('mask_clean', mask_clean)
cv2.imshow('overlay', overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()






