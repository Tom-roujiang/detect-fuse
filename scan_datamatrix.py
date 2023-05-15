import cv2
import os
import img_process
import template

file_path = "./image/detect_datamatrix/"  # 存放待检测图片
file_list = os.listdir(file_path)
for img in file_list:
    img_path = file_path + img
    image = cv2.imread(img_path, -1)
    width = image.shape[0]
    length = image.shape[1]

    """裁剪出大致区域，然后进行霍夫变换，旋转，获取datamatrix"""
    image_crop = image[int(width/4):int(3 * width/4), int(2 * length/5):int(4 * length/5)]
    img_process.hough(image_crop)
    angle = img_process.hough(image_crop)
    img_rotated = img_process.rotate(image_crop, angle)
    bbox = template.detect_datamatrix_roi(img_rotated)
    if bbox:
        data_matrix = img_rotated[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        data_matrix = img_process.sharpen(data_matrix, 110, ktype=1)
        # cv2.imshow('1', data_matrix)
        # cv2.waitKey(0)
    else:
        print('Can not detect datamatrix: ', img)
