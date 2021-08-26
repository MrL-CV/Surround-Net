"""To use the code, users should to config detpath, annopath and imagesetfile detpath is the path for 15 result
files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html" search for
PATH_TO_BE_CONFIGURED to config the paths Note, the evaluation is on the large scale images """
import os
import numpy as np
import re
import time
import sys

import datasets.DOTA_devkit.dota_utils as util
import datasets.DOTA_devkit.polyiou as polyiou
import pdb
import math
from multiprocessing import Pool
from functools import partial

## the thresh for nms when merge image
nms_thresh = 0.1


def py_cpu_nms_poly(dets, thresh):
    scores = dets[:, 8]
    polys = []
    areas = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                           dets[i][2], dets[i][3],
                                           dets[i][4], dets[i][5],
                                           dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]  # 将置信度得分从大到小排列

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]  # 取出当前列表中最大的元素所在的索引
        keep.append(i)
        # 抑制与当前最大值IoU>一定阈值的检测框
        for j in range(order.size - 1):  # 每次都要遍历一遍除选出框以外的所有的框
            iou = polyiou.iou_poly(polys[i], polys[order[j + 1]])
            ovr.append(iou)
        ovr = np.array(ovr)

        # print('ovr: ', ovr)
        # print('thresh: ', thresh)
        try:
            if math.isnan(ovr[0]):
                # 程序运行到这里就会暂停
                pdb.set_trace()
        except:
            pass
        # 返回需要保留的检测框的索引
        # 假如有i个检测框被抑制了,剩余n-1-i个检测框被返回,返回的是从第1个开始(不是从0开始)的符合条件的检测框的索引
        inds = np.where(ovr <= thresh)[0]  # 返回tuple但里面每一个数都是np.array
        # print('inds: ', inds)

        order = order[inds + 1]  # 因为拿出1个用来比较所以整体索引+1,每一次筛选后留下的剩余检测框

    return keep


def py_cpu_nms_poly_fast(dets, thresh):
    obbs = dets[:, 0:-1]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
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
        keep.append(i)
        # if order.size == 0:
        #     break
        # 预筛选,先排除一些根本不可能的,再遍历
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        # h_keep_inds = np.where(hbb_ovr == 0)[0]
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]

        # 预筛选之后,用的还是多边形计算IoU
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou
            # ovr.append(iou)
            # ovr_index.append(tmp_order[j])

        # ovr = np.array(ovr)
        # ovr_index = np.array(ovr_index)
        # print('ovr: ', ovr)
        # print('thresh: ', thresh)
        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]  # 留下来的检测框索引,大于阈值的已经被抑制了

        # order_obb = ovr_index[inds]
        # print('inds: ', inds)
        # order_hbb = order[h_keep_inds + 1]

        order = order[inds + 1]  # 每筛选完一次,就需要更新留下来的检测框

        # pdb.set_trace()
        # order = np.concatenate((order_obb, order_hbb), axis=0).astype(np.int)
    return keep


# 水平框IoU
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


# ?
def nmsbynamedict(nameboxdict, nms, thresh):
    nameboxnmsdict = {x: [] for x in nameboxdict}
    for imgname in nameboxdict:
        # print('imgname:', imgname)
        # keep = py_cpu_nms(np.array(nameboxdict[imgname]), thresh)
        # print('type nameboxdict:', type(nameboxnmsdict))
        # print('type imgname:', type(imgname))
        # print('type nms:', type(nms))
        keep = nms(np.array(nameboxdict[imgname]), thresh)
        # print('keep:', keep)
        outdets = []
        # print('nameboxdict[imgname]: ', nameboxnmsdict[imgname])
        for index in keep:
            # print('index:', index)
            outdets.append(nameboxdict[imgname][index])
        nameboxnmsdict[imgname] = outdets
    return nameboxnmsdict


# 还原回真实尺寸的坐标，返回列表形式（共8个点）
def poly2origpoly(poly, x, y, rate):
    origpoly = []
    for i in range(int(len(poly) / 2)):
        tmp_x = float(poly[i * 2] + x) / float(rate)
        tmp_y = float(poly[i * 2 + 1] + y) / float(rate)
        origpoly.append(tmp_x)
        origpoly.append(tmp_y)
    return origpoly


def mergesingle(dstpath, nms, fullname):
    name = util.custombasename(fullname)  # 取文件夹名
    # print('name:', name)
    dstname = os.path.join(dstpath, name + '.txt')
    with open(fullname, 'r') as f_in:
        nameboxdict = {}
        lines = f_in.readlines()
        splitlines = [x.strip().split(' ') for x in lines]
        for splitline in splitlines:
            subname = splitline[0]
            splitname = subname.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            # print('subname:', subname)
            # 正则表达式:可以用来检查一个串是否含有某种子串、将匹配的子串替换或者从某个串中取出符合某个条件的子串等
            x_y = re.findall(pattern1, subname)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])

            pattern2 = re.compile(r'__([\d+.]+)__\d+___')

            rate = re.findall(pattern2, subname)[0]

            confidence = splitline[1]
            poly = list(map(float, splitline[2:]))
            origpoly = poly2origpoly(poly, x, y, rate)
            det = origpoly
            det.append(float(confidence))
            det = list(map(float, det))
            if oriname not in nameboxdict:
                nameboxdict[oriname] = []
            nameboxdict[oriname].append(det)
        nameboxnmsdict = nmsbynamedict(nameboxdict, nms, nms_thresh)
        with open(dstname, 'w') as f_out:
            for imgname in nameboxnmsdict:
                for det in nameboxnmsdict[imgname]:
                    # print('det:', det)
                    confidence = det[-1]
                    bbox = det[0:-1]
                    outline = imgname + ' ' + str(confidence) + ' ' + ' '.join(map(str, bbox))
                    # print('outline:', outline)
                    f_out.write(outline + '\n')


# 多线程就是多个-单合并同时进行
def mergebase_parallel(srcpath, dstpath, nms):
    # 开多线程
    pool = Pool(16)
    filelist = util.GetFileFromThisRootDir(srcpath)
    # partial包装单合并函数
    mergesingle_fn = partial(mergesingle, dstpath, nms)
    # pdb.set_trace()
    # mergesingle需要3个参数,先用partial解决2个参数,再用map解决最后1个参数!
    # 并行计算的内容是filelist中的所有图像
    pool.map(mergesingle_fn, filelist)


# 合并base:单个合并每张图片
def mergebase(srcpath, dstpath, nms):
    filelist = util.GetFileFromThisRootDir(srcpath)  # 获取srcpath的完整路径
    # 如果不使用多线程,需要用for去遍历待处理图像集
    for filename in filelist:
        mergesingle(dstpath, nms, filename)


# 水平框合并
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


# 旋转框合并
# 多线程+fast合并
def mergebypoly(srcpath, dstpath):
    """
    srcpath: result files before merge and nms
    dstpath: result files after merge and nms
    """
    # srcpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/comp4_test_results'
    # dstpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/testtime'

    # mergebase(srcpath,
    #           dstpath,
    #           py_cpu_nms_poly)
    mergebase_parallel(srcpath,
                       dstpath,
                       py_cpu_nms_poly_fast)


if __name__ == '__main__':
    # 为了能够在win系统上运行,需要将调用作为脚本调用,而不是import调用
    mergebypoly(r'path_to_configure', r'path_to_configure')
    # mergebyrec()
