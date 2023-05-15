import cv2
import os
import numpy as np

np.seterr(divide='ignore', invalid='ignore')


def gamma_trans(img, gamma):  # gamma大于1时图片变暗，小于1图片变亮
    # 具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    # 实现映射用的是Opencv的查表函数
    return cv2.LUT(img, gamma_table)


# 明亮
def Brighter(image, percetage=1.1):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get brighter
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percetage), a_max=255, a_min=0)
    return image_copy


# 调整图片输出大小
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    # print(frame.shape)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


# 模板匹配
def detectobject(img_gray, template, threshold=0.2):
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_gray, pt, (pt[0] + w, pt[1] + h), (255, 255, 255), 1)

    return img_gray


# 图片旋转
def rotate(img, angle=0):
    if len(img.shape) == 3:
        # print(img.shape)
        rows, cols, ch = img.shape  # cols-1 和 rows-1 是坐标限制
        M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), angle, 2)
        dst = cv2.warpAffine(img, M, (cols, rows))  # 意为仿射变换，其中img为输入图像，M为变换矩阵，dsize为图像大小

    if len(img.shape) == 2:
        rows, cols = img.shape  # cols-1 和 rows-1 是坐标限制
        # print(img.shape)
        M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), angle, 1)  # 第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
        dst = cv2.warpAffine(img, M, (cols, rows))  # 意为仿射变换，其中img为输入图像，M为变换矩阵，dsize为图像大小
        # print(M)
    return dst


# 饱和度调整
def PSAlgorithm(rgb_img, increment):
    img = rgb_img * 1.0
    img_min = img.min(axis=2)  # b里面的最小值
    img_max = img.max(axis=2)  # b里面的最大值
    img_out = img
    # 获取HSL空间的饱和度和亮度
    delta = (img_max - img_min) / 255.0
    value = (img_max + img_min) / 255.0
    L = value / 2.0  # L代表亮度
    # s = L<0.5 ? s1 : s2                  # s代表饱和度
    mask_1 = L < 0.5  # mask1 像素中位数
    s1 = delta / (value)  # 对比度=(亮度最大值-亮度最小值)/(亮度最大值+亮度最小值)
    s2 = delta / (2 - value)
    s = s1 * mask_1 + s2 * (1 - mask_1)  # ？？？
    # 增量大于0，饱和度指数增强

    if increment >= 0:
        # alpha = increment+s > 1 ? alpha_1 : alpha_2
        temp = increment + s
        mask_2 = temp > 1
        alpha_1 = s
        alpha_2 = s * 0 + 1 - increment
        alpha = alpha_1 * mask_2 + alpha_2 * (1 - mask_2)
        alpha = 1 / alpha - 1
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha
        img_out = img_out / 255.0
        # 增量小于0，饱和度线性衰减

    else:
        alpha = increment
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha
        img_out = img_out / 255.0
        # RGB颜色上下限处理(小于0取0，大于1取1)
        mask_3 = img_out < 0
        mask_4 = img_out > 1
        img_out = img_out * (1 - mask_3)
        img_out = img_out * (1 - mask_4) + mask_4

    return img_out


# 霍夫变换
def hough(img):
    # img = cv2.GaussianBlur(img, (3,3), 0)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 118)  # 这里对最后一个参数使用了经验型的值
    result = img.copy()
    for line in lines[0]:
        rho = line[0]  # 第一个元素是距离rho
        theta = line[1]  # 第二个元素是角度theta
        if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
            # 该直线与第一行的交点
            pt1 = (int(rho / np.cos(theta)), 0)
            # 该直线与最后一行的交点
            pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
            # 绘制一条白线
            # result = cv2.line(result, pt1, pt2, (255, 255, 255), 3)
            # cv2.imshow('result', rescaleFrame(result, 0.2))
            # cv2.waitKey(0)
            distance = pow(pow(pt1[0] - pt2[0], 2) + pow(pt1[1] - pt2[1], 2), 0.5)
            alpha = np.arccos((pt2[0] - pt1[0]) / distance)
            alpha = 180 * alpha / np.pi
            if alpha > 90:
                alpha -= 90

        else:  # 水平直线
            # 该直线与第一列的交点
            pt1 = (0, int(rho / np.sin(theta)))
            # 该直线与最后一列的交点
            pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
            # 绘制一条直线
            # result = cv2.line(result, pt1, pt2, (255, 255, 255), 3)
            # cv2.imshow('result', rescaleFrame(result, 0.2))
            # cv2.waitKey(0)
            distance = pow(pow(pt1[0] - pt2[0], 2) + pow(pt1[1] - pt2[1], 2), 0.5)
            alpha = np.arccos((pt2[0] - pt1[0]) / distance)
            alpha = 180 * alpha / np.pi
            if alpha > 90:
                alpha -= 90
        return alpha


# 锐化
def sharpen(img, sharpness=100, ktype=1):
    n = sharpness / 100
    if ktype == 1:
        sharpen_op = np.array([[0, -n, 0],\
                               [-n, 4 * n + 1, -n],\
                               [0, -n, 0]], dtype=np.float32)
    if ktype == 2:
        sharpen_op = np.array([[-n, -n, -n],\
                               [-n, 8 * n + 1, -n],\
                               [-n, -n, -n]], dtype=np.float32)
    img_sharpen = cv2.filter2D(img, cv2.CV_32F, sharpen_op)
    img_sharpen = cv2.convertScaleAbs(img_sharpen)
    return img_sharpen
