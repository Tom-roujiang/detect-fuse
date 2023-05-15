import os
import cv2
import numpy as np
import img_process
import template

if __name__ == '__main__':

    file_path = "./image/detect_area/"           # 存放待检测图片
    file_list = os.listdir(file_path)
    size = 2  # size 的值决定视图放大或缩小的大小
    for file in file_list:
        pic = cv2.imread(file_path+file, -1)
        # pic = img_process.gamma_trans(pic, 0.9)
        # pic = img_process.sharpen(pic, sharpness=110, ktype=1)      # 可以尝试做gamma变换和锐化
        pic = cv2.GaussianBlur(pic, (3, 3), 0)

        # !这个操作降分辨率
        pic = img_process.rescaleFrame(pic, 0.3 * size)
        pic_crop1 = pic[250 * size:730 * size, 450 * size:1100 * size]  # pic_crop1 截取保险丝盒大致位置

        '''这段代码用于提取出检验区域的rgb图像'''
        angle = img_process.hough(pic_crop1)
        # bbox_area = template.detect_roi(pic_rotated)
        # edge_crop = pic_rotated[bbox_area[1]:bbox_area[1] + bbox_area[3], bbox_area[0]:bbox_area[0] + bbox_area[2]]
        # edge_crop为最终截取的保险丝盒区域（rgb图片）

        ''' 这段代码用于生成检测区域的边缘二值化图像 '''
        edge = cv2.Canny(pic_crop1, 20, 120, apertureSize=3, L2gradient=False)
        edge = img_process.rotate(edge, angle)
        edge_bbox = template.detect_roi(edge)
        edge = edge[edge_bbox[1]:edge_bbox[1] + edge_bbox[3], edge_bbox[0]:edge_bbox[0] + edge_bbox[2]]
        # edge为检测区域的二值化图像

        '''输出整张图片的检测结果'''
        result = template.detect_fuse(edge)[0]  # result为整张图片的检测结果
        # cv2.imshow(str(num), result)
        # cv2.waitKey(0)

        '''输出每一行每个位置的检测结果'''
        fuse_detection_list = []           # 用于存放检测出的每个位置具体保险丝型号
        fuse_line1 = template.detect_fuse_line1(edge)
        fuse_line2 = template.detect_fuse_line2(edge)
        fuse_line3 = template.detect_fuse_line3(edge)
        fuse_line4 = template.detect_fuse_line4(edge)
        fuse_detection_list.append(fuse_line1)
        fuse_detection_list.append(fuse_line2)
        fuse_detection_list.append(fuse_line3)
        fuse_detection_list.append(fuse_line4)
        matrix = template.change_list_to_matrix(fuse_detection_list)  # 输出0,1矩阵
        print(fuse_detection_list)
        # print(matrix)

        fuse_template_list = [['20', '5', '0', '10', '30', '20', '10', '2', '0', '0', '5', '15', '15', '5'],
                              ['10', '10', '5', '5', '0', '30', '0', '0'],
                              ['5', '15', '30', '10'],
                              ['5', '0', '20', '10']]

        fuse_matrix_list = [[1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                            [1, 1, 1, 1, 0, 1, 0, 0],
                            [1, 1, 1, 1],
                            [1, 0, 1, 1]]

        # template.compare_list(fuse_detection_list, fuse_template_list, img_path_list[num])      # 比较每个位置上保险丝和模板的匹配程度
        template.compare_list(matrix, fuse_matrix_list, file)         # 比较是否漏装
