"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import os
import numpy as np
import dota_utils as util
import re
import time
import polyiou

## the thresh for nms when merge image
nms_thresh = 0.3


def py_cpu_nms_poly(dets, thresh):
    # 	取出每个检测框的得分
    scores = dets[:, 8]
    polys = []
    areas = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                           dets[i][2], dets[i][3],
                                           dets[i][4], dets[i][5],
                                           dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        # 		取出最高得分者
        keep.append(i)
        # 		计算所有检测框与它的iou值
        for j in range(order.size - 1):
            iou = polyiou.iou_poly(polys[i], polys[order[j + 1]])
            ovr.append(iou)
        ovr = np.array(ovr)
        # 		保留小于=阈值的检测框
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep  # 返回的keep是检测框的原始索引


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    # print('dets:', dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ## index for dets
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def nmsbynamedict(nameboxdict, nms, thresh):
    # 	列表化
    nameboxnmsdict = {x: [] for x in nameboxdict}
    # 	遍历每个图片名字
    for imgname in nameboxdict:
        # 		该图片下保留下来的检测框
        keep = nms(np.array(nameboxdict[imgname]), thresh)
        outdets = []
        # 		遍历保留下来的检测框
        for index in keep:
            # 			将其放入同一个图像名字下
            outdets.append(nameboxdict[imgname][index])
        # 		对每幅图都这么做，因为是本地算法，所以不会创建副本而造成资源占用
        nameboxnmsdict[imgname] = outdets
    return nameboxnmsdict


# 还原回真实尺寸的坐标，返回列表形式（共8个点）
def poly2origpoly(poly, x, y, rate):
    origpoly = []
    for i in range(int(len(poly) / 2)):
        #         不过只需要还原2个方面：①还原平移；②还原缩放比例
        tmp_x = float(poly[i * 2] + x) / float(rate)
        tmp_y = float(poly[i * 2 + 1] + y) / float(rate)
        origpoly.append(tmp_x)
        origpoly.append(tmp_y)
    return origpoly


# 合并检测标签
def mergebase(srcpath, dstpath, nms):
    # 	取文件开始遍历
    filelist = util.GetFileFromThisRootDir(srcpath)
    for fullname in filelist:
        name = util.custombasename(fullname)
        # 		目标文件的.txt名字
        dstname = os.path.join(dstpath, name + '.txt')
        # 		读取离散检测标签
        with open(fullname, 'r') as f_in:
            nameboxdict = {}
            # 			一口气读完
            lines = f_in.readlines()
            # 			再按照空格分隔
            splitlines = [x.strip().split(' ') for x in lines]
            # 			逐个读取标签
            for splitline in splitlines:
                subname = splitline[0]
                splitname = subname.split('__')
                # 				图片名字（P2075等）
                oriname = splitname[0]
                pattern1 = re.compile(r'__\d+___\d+')
                x_y = re.findall(pattern1, subname)  # 返回匹配上的字符串，并放到列表中
                # 				只取出数字，也就是当前patch的左坐标与上坐标
                x_y_2 = re.findall(r'\d+', x_y[0])
                # 				取出具体左上值
                x, y = int(x_y_2[0]), int(x_y_2[1])

                # 				取出rate
                pattern2 = re.compile(r'__([\d+\.]+)__\d+___')
                rate = re.findall(pattern2, subname)[0]
                # 				[0]是名字，[1]是置信度
                confidence = splitline[1]
                # 				将坐标转换为浮点并形成列表
                poly = list(map(float, splitline[2:]))
                # 				得到原始尺寸下的坐标（列表形式,8个坐标）
                #               *这个步骤统一了不同rate下的检测结果
                origpoly = poly2origpoly(poly, x, y, rate)
                #                 print("到底是什么？",origpoly)
                det = origpoly
                det.append(confidence)
                # 				全都将数转换为浮点
                det = list(map(float, det))
                # 				如果第一次见这个图片的内容
                if (oriname not in nameboxdict):
                    nameboxdict[oriname] = []
                # 				向这个图片里添加检测结果
                nameboxdict[oriname].append(det)
            # 			同一个合并文件下的NMS
            nameboxnmsdict = nmsbynamedict(nameboxdict, nms, nms_thresh)
            with open(dstname, 'w') as f_out:
                # 				遍历总图片
                for imgname in nameboxnmsdict:
                    # 					遍历每张图片的检测结果
                    for det in nameboxnmsdict[imgname]:
                        confidence = det[-1]
                        bbox = det[0:-1]
                        outline = imgname + ' ' + str(confidence) + ' ' + ' '.join(map(str, bbox))
                        #                         写入最后的文件
                        f_out.write(outline + '\n')


def mergebyrec(srcpath, dstpath):
    """
    srcpath: result files before merge and nms
    dstpath: result files after merge and nms
    """
    # srcpath = r'E:\bod-dataset\results\bod-v3_rfcn_2000000'
    # dstpath = r'E:\bod-dataset\results\bod-v3_rfcn_2000000_nms'

    mergebase(srcpath,
              dstpath,
              py_cpu_nms)


def mergebypoly(srcpath, dstpath):
    """
    srcpath: result files before merge and nms
    dstpath: result files after merge and nms
    """
    # srcpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/comp4_test_results'
    # dstpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/testtime'

    temp_dir = os.path.normpath(dstpath)

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    mergebase(srcpath,
              dstpath,
              py_cpu_nms_poly)


if __name__ == '__main__':
    # see demo for example
    mergebypoly(r'path_to_configure', r'path_to_configure')
    # mergebyrec()