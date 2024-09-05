import pdb
import cv2
import numpy as np
import matplotlib.pyplot as plt

def homography_align(img1, img2):
    # 转换为灰度图像
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 使用 ORB 检测关键点和描述符
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # 使用 BFMatcher 进行特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 提取匹配点
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 计算 Homography 矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 使用 Homography 矩阵对图像进行变换
    height, width, channels = img2.shape
    aligned_img = cv2.warpPerspective(img1, H, (width, height), borderMode=cv2.BORDER_REPLICATE)
    
    return aligned_img