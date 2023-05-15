import os
from MTM import matchTemplates, drawBoxesOnRGB
import cv2
import numpy as np

roi_temp_path = './template_image/template_roi/0.jpg'     # roi区域模板图
datamatrix_temp_path = './template_image/template_roi/datamatrix_template.jpg'      # datamatrix的template
fuse_temp_path = './template_image/template_fuse/'
temp_roi = cv2.imread(roi_temp_path, 0)
datamatrix_temp_roi = cv2.imread(datamatrix_temp_path, 0)

def readTemplates(temp_path):
    """用于读取检测的保险丝模板"""
    listTemplate = []
    folder_list = os.listdir(temp_path)
    for folder in folder_list:
        folder_path = os.path.join(temp_path, folder + '/')
        file_list = os.listdir(folder_path)
        for file in file_list:
            filepath = os.path.join(folder_path, file)
            # print(filepath)
            temp = cv2.imread(filepath, 0)  # 读取每个模板
            listTemplate.append((folder, temp))
    return listTemplate


def detect_fuse(image):
    """用于检测图片中的保险丝"""
    listTemplate = readTemplates(fuse_temp_path)
    for i, index in enumerate([2, 5, 10, 15, 20, 30]):
        if index == 30:
            for k_num in range(4):
                rotated = np.rot90(listTemplate[i][1], k=k_num)  # NB: np.rotate not good here, turns into float!
                listTemplate.append((str(index), rotated))
        else:
            rotated = np.rot90(listTemplate[i][1], k=2)  # NB: np.rotate not good here, turns into float!
            listTemplate.append((str(index), rotated))
    # score_threshold 用于设置置信度
    Hits = matchTemplates(listTemplate, image, score_threshold=0.45, method=cv2.TM_CCOEFF_NORMED, maxOverlap=0.3)
    Overlay = drawBoxesOnRGB(image, Hits, showLabel=True)
    return Overlay, Hits  # 返回输出图片和结果信息


def detect_roi(image):
    """用于检测图片中的保险丝盒区域"""
    listTemplate = [('roi', temp_roi)]
    Hits = matchTemplates(listTemplate, image, score_threshold=0.03, method=cv2.TM_CCOEFF_NORMED, maxOverlap=0.2)
    bbox = Hits['BBox'][0]
    # Overlay = drawBoxesOnRGB(image, Hits, showLabel=True)
    return bbox


def detect_datamatrix_roi(image):
    """用于检测图片中的二维码区域"""
    listTemplate = [('roi', datamatrix_temp_roi)]
    Hits = matchTemplates(listTemplate, image, score_threshold=0.3, method=cv2.TM_CCOEFF_NORMED, maxOverlap=0.2)
    try:
        bbox = Hits['BBox'][0]
        Overlay = drawBoxesOnRGB(image, Hits, showLabel=True)
    except IndexError:
        bbox = ()
    return bbox


def detect_fuse_line1(image):
    """用于检测图片中的保险丝盒第1行"""
    width = image.shape[0]
    length = image.shape[1]
    length_unit = int(length / 15)
    fuse_list = []
    for i in range(14):
        result_line1 = image[0:int(width / 3.5), int(length_unit * (i + 0.3)): int(length_unit * (i + 1.7))]
        result, result_info = detect_fuse(result_line1)
        if result_info.empty == False:
            '''tolist 可以不带索引名输出'''
            if len(result_info.loc[:, 'TemplateName'].tolist()) == 1:
                fuse_list.append(result_info.loc[:, 'TemplateName'].tolist()[0])  # 区域内只有一个保险丝
            else:
                print('Error! ' + str(len(result_info.loc[:, 'TemplateName'].tolist())) + ' outcomes in location line1_' + str(i + 1))  # 区域内有多个保险丝
                print(result_info.loc[:, 'TemplateName'].tolist())
                fuse_list.append('X')
        else:
            fuse_list.append('0')

    return fuse_list


def detect_fuse_line2(image):
    """用于检测图片中的保险丝盒第2行"""
    width = image.shape[0]
    length = image.shape[1]
    length_unit = int(length / 15)
    fuse_list = []
    for i in range(4):
        result_line2 = image[int(width / 4): int(2* width/4), int(length_unit * (i + 0.3)): int(length_unit * (i + 1.7))]
        result, result_info = detect_fuse(result_line2)
        # cv2.imshow(str(i), result)
        # cv2.waitKey(0)
        if result_info.empty == False:
            '''tolist 可以不带索引名输出'''
            if len(result_info.loc[:, 'TemplateName'].tolist()) == 1:
                fuse_list.append(result_info.loc[:, 'TemplateName'].tolist()[0])  # 区域内只有一个保险丝
            else:
                print('Error! ' + str(len(result_info.loc[:, 'TemplateName'].tolist())) + 'outcomes in location '
                                                                                          'line2_' + str(i + 1))
                # 区域内有多个保险丝
                print(result_info.loc[:, 'TemplateName'].tolist())
                fuse_list.append('X')
        else:
            fuse_list.append('0')

    for j in range(4, 8):
        length_unit_big = int(length / 9)
        result_line2 = image[int(width / 4): int(2 * width / 4),
                       int(length_unit_big * (j - 1.3)): int(length_unit_big * (j - 0.3))]
        result, result_info = detect_fuse(result_line2)
        # cv2.imshow(str(j), result)
        # cv2.waitKey(0)
        if result_info.empty == False:
            '''tolist 可以不带索引名输出'''
            temp_fuse_list = result_info.loc[:, 'TemplateName'].tolist()  # 存放具有两个30的大块保险丝标签
            # print(temp_fuse_list)
            if len(temp_fuse_list) == 1:
                fuse_list.append(temp_fuse_list[0])  # 区域内只有一个保险丝
            elif len(temp_fuse_list) == 2 and temp_fuse_list[0] == '30':
                # print(temp_fuse_list)
                fuse_list.append(temp_fuse_list[0])  # 区域内有两个保险丝，但是都是30
            else:
                print('Error! ' + str(len(result_info.loc[:, 'TemplateName'].tolist())) + 'outcomes in location '
                                                                                          'line2_' + str(j + 1))  #
                # 区域内有多个保险丝
                print(temp_fuse_list)
                cv2.imshow('temp', result)
                cv2.waitKey(0)
                fuse_list.append('X')
        else:
            fuse_list.append('0')

    return fuse_list


def detect_fuse_line3(image):
    """用于检测图片中的保险丝盒第3行"""
    width = image.shape[0]
    length = image.shape[1]
    length_unit = int(length / 15)
    fuse_list = []
    r = [range(0, 2), range(6, 8)]
    for small_range in r:
        for i in small_range:
            result_line3 = image[int(2 * width / 4):int(3 * width / 4), int(length_unit * (i + 0.3)): int(length_unit * (i + 1.7))]
            result, result_info = detect_fuse(result_line3)
            # cv2.imshow(str(i),result)
            # cv2.waitKey(0)
            if result_info.empty == False:
                '''tolist 可以不带索引名输出'''
                if len(result_info.loc[:, 'TemplateName'].tolist()) == 1:
                    fuse_list.append(result_info.loc[:, 'TemplateName'].tolist()[0])  # 区域内只有一个保险丝
                else:
                    print('Error! ' + str(len(result_info.loc[:, 'TemplateName'].tolist())) + ' outcomes in location line3_' + str(i + 1))  # 区域内有多个保险丝
                    print(result_info.loc[:, 'TemplateName'].tolist())

                    fuse_list.append('X')
            else:
                fuse_list.append('0')

    return fuse_list


def detect_fuse_line4(image):
    """用于检测图片中的保险丝盒第4行"""
    width = image.shape[0]
    length = image.shape[1]
    length_unit = int(length / 15)
    fuse_list = []
    r = [range(0, 2), range(6, 8)]
    for small_range in r:
        for i in small_range:
            result_line4 = image[int(2.8 * width / 4): width, int(length_unit * (i + 0.3)): int(length_unit * (i + 1.7))]
            result, result_info = detect_fuse(result_line4)
            # cv2.imshow(str(i),result)
            # cv2.waitKey(0)
            if result_info.empty == False:
                '''tolist 可以不带索引名输出'''
                if len(result_info.loc[:, 'TemplateName'].tolist()) == 1:
                    fuse_list.append(result_info.loc[:, 'TemplateName'].tolist()[0])  # 区域内只有一个保险丝
                else:
                    print('Error! ' + str(len(result_info.loc[:, 'TemplateName'].tolist())) + ' outcomes in location line4_' + str(i + 1))  # 区域内有多个保险丝
                    print(result_info.loc[:, 'TemplateName'].tolist())

                    fuse_list.append('X')
            else:
                fuse_list.append('0')

    return fuse_list


def change_list_to_matrix(temp_fuse_list):
    """用于将检测结果转换为包含0和1的矩阵"""
    fuse_matrix = []
    fuse_matrix_line1 = []
    fuse_matrix_line2 = []
    fuse_matrix_line3 = []
    fuse_matrix_line4 = []
    for i in range(14):
        if temp_fuse_list[0][i] != '0':
            fuse_matrix_line1.append(1)
        else:
            fuse_matrix_line1.append(0)
    for j in range(8):
        if temp_fuse_list[1][j] != '0':
            fuse_matrix_line2.append(1)
        else:
            fuse_matrix_line2.append(0)
    for k in range(4):
        if temp_fuse_list[2][k] != '0':
            fuse_matrix_line3.append(1)
        else:
            fuse_matrix_line3.append(0)
    for l in range(4):
        if temp_fuse_list[3][l] != '0':
            fuse_matrix_line4.append(1)
        else:
            fuse_matrix_line4.append(0)
    fuse_matrix.append(fuse_matrix_line1)
    fuse_matrix.append(fuse_matrix_line2)
    fuse_matrix.append(fuse_matrix_line3)
    fuse_matrix.append(fuse_matrix_line4)
    return fuse_matrix


def compare_list(detect_list, template_list, file):
    if detect_list == template_list:
        print(file, 'True')
    else:
        print(file, 'False')


