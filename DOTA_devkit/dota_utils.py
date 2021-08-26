import sys
import codecs
import numpy as np
import shapely.geometry as shgeo
import os
import re
import math

# import polyiou
"""
    some basic functions which are useful for process DOTA data
"""

# 字符串数组,全局变量
wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
               'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool',
               'helicopter']


def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])


# 获取所需文件的完整路径
def GetFileFromThisRootDir(dir, ext=None):
    #     文件名结合列表
    allfiles = []
    # ext是额外文件夹?
    needExtFilter = (ext != None)
    #     返回:当前文件夹路径,当前文件夹所有文件夹的名字,当前文件夹中的所有文件
    #     print(dir)
    for root, dirs, files in os.walk(dir):
        print("遍历当前文件夹得到的文件名称:", files)
        for filespath in files:
            #         组合路径得到每一个文件的单独路径
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]  # 这里取1代表只看后缀,第二个1代表不要.号
            #     如果需要额外的文件:标志位打开 + 拓展名在所输入的拓展名中
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            #         标志位没打开
            elif not needExtFilter:
                allfiles.append(filepath)
    #     但标志位打开但拓展名却不在输入的拓展名中的文件不加入文件名集合中
    return allfiles


# 将元组拆开，合并放在一个列表中
def TuplePoly2Poly(poly):
    outpoly = [poly[0][0], poly[0][1],
               poly[1][0], poly[1][1],
               poly[2][0], poly[2][1],
               poly[3][0], poly[3][1]
               ]
    return outpoly


# 解析坐标,返回字典
# 返回以下key:'name', 'difficult', 'poly', 'area'
# 输入:一个标签文件的路径
def parse_dota_poly(filename):
    """
        parse the dota ground truth in the format:
        [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    objects = []
    # print('filename:', filename)
    f = []
    #     检查python版本,并打开文件
    if sys.version_info >= (3, 5):
        fd = open(filename, 'r')
        f = fd
    elif sys.version_info >= (2, 7):
        fd = codecs.open(filename, 'r')
        f = fd
    # count = 0
    while True:
        line = f.readline()
        # count = count + 1
        # if count < 2:
        #     continue
        if line:  # 如果读取到东西
            splitlines = line.strip().split(' ')  # 按照空格分割这一行的读取数据
            object_struct = {}

            if len(splitlines) < 9:  # 分割数<9的肯定不是数据
                continue
            if len(splitlines) >= 9:
                object_struct['name'] = splitlines[8]
            if len(splitlines) == 9:  # 困难值没有，设置0给这个目标
                object_struct['difficult'] = '0'
            elif len(splitlines) >= 10:
                object_struct['difficult'] = splitlines[9]

            object_struct['poly'] = [(float(splitlines[0]), float(splitlines[1])),
                                     (float(splitlines[2]), float(splitlines[3])),
                                     (float(splitlines[4]), float(splitlines[5])),
                                     (float(splitlines[6]), float(splitlines[7]))
                                     ]
            gtpoly = shgeo.Polygon(object_struct['poly'])
            object_struct['area'] = gtpoly.area
            #             向列表中加入字典
            objects.append(object_struct)
        else:
            break
    return objects


def parse_dota_poly2(filename):
    """
        parse the dota ground truth in the format:
        [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    objects = parse_dota_poly(filename)
    for obj in objects:
        obj['poly'] = TuplePoly2Poly(obj['poly'])
        obj['poly'] = list(map(int, obj['poly']))  # 再转换为int类型
    return objects


def parse_dota_rec(filename):
    """
        parse the dota ground truth in the bounding box format:
        "xmin, ymin, xmax, ymax"
    """
    objects = parse_dota_poly(filename)
    for obj in objects:
        poly = obj['poly']
        bbox = dots4ToRec4(poly)
        obj['bndbox'] = bbox
    return objects


## bounding box transfer for varies format

def dots4ToRec4(poly):
    xmin, xmax, ymin, ymax = min(poly[0][0], min(poly[1][0], min(poly[2][0], poly[3][0]))), \
                             max(poly[0][0], max(poly[1][0], max(poly[2][0], poly[3][0]))), \
                             min(poly[0][1], min(poly[1][1], min(poly[2][1], poly[3][1]))), \
                             max(poly[0][1], max(poly[1][1], max(poly[2][1], poly[3][1])))
    return xmin, ymin, xmax, ymax


def dots4ToRec8(poly):
    xmin, ymin, xmax, ymax = dots4ToRec4(poly)
    return xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax
    # return dots2ToRec8(dots4ToRec4(poly))


def dots2ToRec8(rec):
    xmin, ymin, xmax, ymax = rec[0], rec[1], rec[2], rec[3]
    return xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax


# 参数：源文件夹->目标文件夹
# 将源文件夹中包含多类的label分成各个类的集合label
def groundtruth2Task1(srcpath, dstpath):
    filelist = GetFileFromThisRootDir(srcpath)

    filedict = {}  # 字典

    TempPath = os.path.normpath(dstpath)
    if not os.path.exists(TempPath):
        os.makedirs(TempPath)

    for cls in wordname_15:
        fd = open(os.path.join(TempPath, 'Task1_') + cls + r'.txt', 'w')
        # 		赋予每一个类都拥有一个.txt文件和读写权限
        filedict[cls] = fd

    # 	遍历读取的文件列表（label文件）
    for filepath in filelist:
        objects = parse_dota_poly2(filepath)
        subname = custombasename(filepath)
        # 		利用正则表达式去进行匹配字符串
        pattern2 = re.compile(r'__([\d+\.]+)__\d+___')
        # 		找到所有匹配的字符串中的缩放系数rate
        rate = re.findall(pattern2, subname)[0]

        for obj in objects:
            category = obj['name']
            difficult = obj['difficult']
            poly = obj['poly']

            # 			如果难度>1跳过
            if difficult == '2':
                continue

            # 			不同比率的resize对应不同的假设score
            if rate == '0.5':
                outline = custombasename(filepath) + ' ' + '1' + ' ' + ' '.join(map(str, poly))
            elif rate == '1':
                outline = custombasename(filepath) + ' ' + '0.8' + ' ' + ' '.join(map(str, poly))
            elif rate == '2':
                outline = custombasename(filepath) + ' ' + '0.6' + ' ' + ' '.join(map(str, poly))
            # 			向保存有特定类别的txt文件中写入坐标与标记信息
            filedict[category].write(outline + '\n')


# 将merge and nms之后的标签信息
def Task2groundtruth_poly(srcpath, dstpath):
    temp_dir = os.path.normpath(dstpath)

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    thresh = 0.1
    filedict = {}
    # 	从合并标签文件夹中读取labels，指定为.txt
    Tasklist = GetFileFromThisRootDir(srcpath, '.txt')

    for Taskfile in Tasklist:
        idname = custombasename(Taskfile).split('_')[-1]  # 去掉Task1和_，只留下具体种类
        f = open(Taskfile, 'r')  # 打开文件
        lines = f.readlines()  # 全文件读取
        # 		判断每一行的有效性
        for line in lines:
            if len(line) == 0:
                continue
            # 			对读取的每一行使用空格分隔，存放在列表splitline中
            splitline = line.strip().split(' ')
            filename = splitline[0]
            confidence = splitline[1]
            bbox = splitline[2:]
            # 			如果置信度>阈值
            if float(confidence) > thresh:
                # 				如果还没有这个种类的文件，就创建一个新的
                if filename not in filedict:
                    # 					使用codecs.open来打开文件，保证了后续不同内容都可以写入成功
                    filedict[filename] = codecs.open(os.path.join(dstpath, filename + '.txt'), 'w')

                poly = bbox
                # 				最后将内容写入txt文件中
                filedict[filename].write(' '.join(poly) + ' ' + idname + '\n')


def polygonToRotRectangle(bbox):
    """
    :param bbox: The polygon stored in format [x1, y1, x2, y2, x3, y3, x4, y4]
    :return: Rotated Rectangle in format [cx, cy, w, h, theta]
    """
    bbox = np.array(bbox, dtype=np.float32)
    bbox = np.reshape(bbox, newshape=(2, 4), order='F')
    angle = math.atan2(-(bbox[0, 1] - bbox[0, 0]), bbox[1, 1] - bbox[1, 0])

    center = [[0], [0]]

    for i in range(4):
        center[0] += bbox[0, i]
        center[1] += bbox[1, i]

    center = np.array(center, dtype=np.float32) / 4.0

    R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)

    normalized = np.matmul(R.transpose(), bbox - center)

    xmin = np.min(normalized[0, :])
    xmax = np.max(normalized[0, :])
    ymin = np.min(normalized[1, :])
    ymax = np.max(normalized[1, :])

    w = xmax - xmin + 1
    h = ymax - ymin + 1

    return [float(center[0]), float(center[1]), w, h, angle]


def cal_line_length(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def get_best_begin_point(coordinate):
    x1 = coordinate[0][0]
    y1 = coordinate[0][1]
    x2 = coordinate[1][0]
    y2 = coordinate[1][1]
    x3 = coordinate[2][0]
    y3 = coordinate[2][1]
    x4 = coordinate[3][0]
    y4 = coordinate[3][1]
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) + cal_line_length(combinate[i][1],
                                                                                           dst_coordinate[
                                                                                               1]) + cal_line_length(
            combinate[i][2], dst_coordinate[2]) + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        print("choose one direction!")
    return combinate[force_flag]
