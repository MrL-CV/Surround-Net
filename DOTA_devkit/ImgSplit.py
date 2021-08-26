import os
import codecs
import numpy as np
import math
from .dota_utils import GetFileFromThisRootDir
import cv2
import shapely.geometry as shgeo
from . import dota_utils as util
import copy


# 参数：5顶点->4顶点坐标,原版完整4顶点坐标
def choose_best_pointorder_fit_another(poly1, poly2):
    """
        To make the two polygons best fit with each point
    """
    x1 = poly1[0]
    y1 = poly1[1]
    x2 = poly1[2]
    y2 = poly1[3]
    x3 = poly1[4]
    y3 = poly1[5]
    x4 = poly1[6]
    y4 = poly1[7]
    #     循环寻找一个最合适的排列顺序：坐标之间差值最小
    combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                 np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
    dst_coordinate = np.array(poly2)
    distances = np.array([np.sum((coord - dst_coordinate) ** 2) for coord in combinate])
    sorted_best = distances.argsort()
    #     返回与原版目标差值最小的排列
    return combinate[sorted_best[0]]


# 计算2个顶点之间的距离
def cal_line_length(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


class splitbase:
    def __init__(self,
                 basepath,  # 存放图片文件夹与标签文件夹的总文件夹
                 outpath,  # 存放输出图片文件夹与标签文件夹的总文件夹
                 code='utf-8',
                 gap=100,  # 分割的子图片之间的重叠率
                 subsize=1024,  # 分割尺寸
                 thresh=0.7,  # 裁剪之后物体的残缺部分如果占比自身太少,忽略那个残缺部分
                 choosebestpoint=True,  # 用于在5顶点->4顶点之后,新插入的坐标的正确排序方式
                 ext='.png'
                 ):
        """
        :param basepath: base path for dota data
        :param outpath: output base path for dota data,
        the basepath and outputpath have the similar subdirectory, 'images' and 'labelTxt'
        :param code: encodeing format of txt file
        :param gap: overlap between two patches
        :param subsize: subsize of patch
        :param thresh: the thresh determine whether to keep the instance if the instance
        is cut down in the process of split
        :param choosebestpoint: used to choose the first point for the
        :param ext: ext for the image format
        """
        self.basepath = basepath
        self.outpath = outpath
        self.code = code
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.thresh = thresh
        #         输入目录
        self.imagepath = os.path.join(self.basepath, 'images')
        self.labelpath = os.path.join(self.basepath, 'labelTxt')
        #         输出目录
        self.outimagepath = os.path.join(self.outpath, 'images')
        self.outlabelpath = os.path.join(self.outpath, 'labelTxt')
        self.choosebestpoint = choosebestpoint
        self.ext = ext
        #         如果输出目录不存在就创建
        if not os.path.exists(self.outimagepath):
            os.makedirs(self.outimagepath)
        if not os.path.exists(self.outlabelpath):
            os.makedirs(self.outlabelpath)

    ## point: (x, y), rec: (xmin, ymin, xmax, ymax)
    # def __del__(self):
    #     self.f_sub.close()
    ## grid --> (x, y) position of grids

    #     获取目标在裁剪图像中的新坐标
    def polyorig2sub(self, left, up, poly):
        polyInsub = np.zeros(len(poly))
        #         i = 0,1,2,3   实际可以索引到的索引：0,2,4,6/1,3,5,7
        for i in range(int(len(poly) / 2)):
            #             对x坐标重新计算新的位置
            polyInsub[i * 2] = int(poly[i * 2] - left)
            #             对y坐标重新计算新的位置
            polyInsub[i * 2 + 1] = int(poly[i * 2 + 1] - up)
        return polyInsub

    def calchalf_iou(self, poly1, poly2):
        """
            It is not the iou on usual, the iou is the value of intersection over poly1
        """
        #         计算（可能截断的）目标面积
        inter_poly = poly1.intersection(poly2)
        inter_area = inter_poly.area
        #         计算（可能截断的）目标面积与完整目标面积的比例
        poly1_area = poly1.area
        half_iou = inter_area / poly1_area
        #         返回阶段部分实例 + 与完整面积的比值
        return inter_poly, half_iou

    #     通过深拷贝进行裁剪实现
    def saveimagepatches(self, img, subimgname, left, up):
        subimg = copy.deepcopy(img[up: (up + self.subsize), left: (left + self.subsize)])
        outdir = os.path.join(self.outimagepath, subimgname + self.ext)
        cv2.imwrite(outdir, subimg)

    #     中位数插值法从5顶点多边形获取4顶点矩形
    def GetPoly4FromPoly5(self, poly):
        #         计算两个相邻顶点的线段距离
        distances = [cal_line_length((poly[i * 2], poly[i * 2 + 1]), (poly[(i + 1) * 2], poly[(i + 1) * 2 + 1])) for i
                     in range(int(len(poly) / 2 - 1))]
        #         补充计算头顶点和尾顶点的距离
        distances.append(cal_line_length((poly[0], poly[1]), (poly[8], poly[9])))
        #         将distances中的元素从小到大排列，提取其对应的index(索引)，然后pos=最小值的索引
        pos = np.array(distances).argsort()[0]
        count = 0
        outpoly = []
        while count < 5:
            # print('count:', count)
            #             如果来到了最小那条边所在的顶点，取其与下一个顶点的中点作为新的顶点
            if count == pos:
                outpoly.append((poly[count * 2] + poly[(count * 2 + 2) % 10]) / 2)
                outpoly.append((poly[(count * 2 + 1) % 10] + poly[(count * 2 + 3) % 10]) / 2)
                count = count + 1

            #             因为被与上一个最小边所在的顶点合并，所以下一个顶点被忽略
            elif count == (pos + 1) % 5:
                count = count + 1
                continue

            #             正常计数，正常添加数据（添加的是新裁剪图中的顶点坐标）
            else:
                outpoly.append(poly[count * 2])
                outpoly.append(poly[count * 2 + 1])
                count = count + 1
        return outpoly

    #     保存单次裁剪的结果（处理标签部分）
    def savepatches(self, resizeimg, objects, subimgname, left, up, right, down):
        #         裁剪块的输出标签名字
        outdir = os.path.join(self.outlabelpath, subimgname + '.txt')
        #         mask_poly = []
        #         构建一个多边形对象
        imgpoly = shgeo.Polygon([(left, up), (right, up), (right, down), (left, down)])
        #         按照'utf-8'编码的读写方式打开输出txt文件
        with codecs.open(outdir, 'w', self.code) as f_out:
            #             读取标签中每个目标的检测框区域
            for obj in objects:
                gtpoly = shgeo.Polygon([(obj['poly'][0], obj['poly'][1]),
                                        (obj['poly'][2], obj['poly'][3]),
                                        (obj['poly'][4], obj['poly'][5]),
                                        (obj['poly'][6], obj['poly'][7])])
                #                 如果resize后太小，就丢弃
                if gtpoly.area <= 0:
                    continue
                #                 获取（可能截断）的目标面积与完整目标面积比
                inter_poly, half_iou = self.calchalf_iou(gtpoly, imgpoly)

                # print('writing...')
                #                 如果没有被截断
                if half_iou == 1:
                    #                     将目标坐标的新信息写到对应的标签文件中
                    polyInsub = self.polyorig2sub(left, up, obj['poly'])
                    #                     使用空格填充，例子：https://blog.csdn.net/qq_38786209/article/details/78304974
                    outline = ' '.join(list(map(str, polyInsub)))
                    #                     添加该目标的其他信息
                    outline = outline + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    #                     写入文件中
                    f_out.write(outline + '\n')
                #                 如果被截断，获取截断部分的环绕点（按逆时针）
                elif half_iou > 0:
                    # elif (half_iou > self.thresh):
                    ##  print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                    inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                    out_poly = list(inter_poly.exterior.coords)[0: -1]

                    #                     如果截断成了一个三角形，不处理，读取下一个目标
                    if len(out_poly) < 4:
                        continue

                    #                     如果顶点个数>=4，添加顶点信息到列表中
                    out_poly2 = []
                    for i in range(len(out_poly)):
                        out_poly2.append(out_poly[i][0])
                        out_poly2.append(out_poly[i][1])

                    #                     如果顶点数==5，有专门函数处理
                    if len(out_poly) == 5:
                        # print('==========================')
                        out_poly2 = self.GetPoly4FromPoly5(out_poly2)

                    #                     如果顶点>5，忽略
                    elif len(out_poly) > 5:
                        """
                            if the cut instance is a polygon with points more than 5, we do not handle it currently
                        """
                        continue

                    #                     使用最佳排列
                    if self.choosebestpoint:
                        out_poly2 = choose_best_pointorder_fit_another(out_poly2, obj['poly'])
                    #                     因为之前只是对没有被截断的目标进行坐标重新定位
                    polyInsub = self.polyorig2sub(left, up, out_poly2)

                    #                     检查新坐标合法性
                    for index, item in enumerate(polyInsub):
                        #                         <=1的置为1
                        if item <= 1:
                            polyInsub[index] = 1
                        #                         >=边界的置为边界
                        elif item >= self.subsize:
                            polyInsub[index] = self.subsize
                    #                     同上面未被截断的情况一样
                    outline = ' '.join(list(map(str, polyInsub)))
                    #                     如果裁剪后目标太小，就在训练中忽略
                    if half_iou > self.thresh:
                        outline = outline + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    else:
                        ## if the left part is too small, label as '2'
                        outline = outline + ' ' + obj['name'] + ' ' + '2'
                    f_out.write(outline + '\n')
                # else:
                #   mask_poly.append(inter_poly)
        #         标签处理完之后开始保存裁剪的图片
        self.saveimagepatches(resizeimg, subimgname, left, up)

    #         切割单张图片
    def SplitSingle(self, name, rate, extent):
        """
            split a single image and ground truth
        :param name: image name
        :param rate: the resize scale for the image
        :param extent: the image format
        :return:
        """
        #     读图
        img = cv2.imread(os.path.join(self.imagepath, name + extent))
        #     没读到图，退出
        if np.shape(img) == ():
            return

        #     读标签（读整个图像的所有标签）
        fullname = os.path.join(self.labelpath, name + '.txt')
        #     解析坐标，返回字典
        objects = util.parse_dota_poly2(fullname)

        #     对坐标下手，随图片的缩放程度一起改变，并将坐标信息改为：[x1, y1, x2, y2, x3, y3, x4, y4]形式，其他标签不变
        for obj in objects:
            #             利用map完成坐标更新的计算
            obj['poly'] = list(map(lambda x: rate * x, obj['poly']))
        #      如果缩放比例！=1，说明不是原图
        if rate != 1:
            #             通过宽和高的比例分别指定进行resize
            resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)  # 4x4像素邻域的双三次插值
        else:
            resizeimg = img
        #      输出切割图片的名字
        outbasename = name + '__' + str(rate) + '__'
        #      resize后图片的宽、高
        weight = np.shape(resizeimg)[1]
        height = np.shape(resizeimg)[0]

        #      切割的原则：左边界和上边界在滑动，滑动距离是子区域->重叠距离
        #      下边界和右边界负责检测是否到达边界，到达边界取边界
        left, up = 0, 0
        while left < weight:
            if left + self.subsize >= weight:
                left = max(weight - self.subsize, 0)
            up = 0
            while up < height:
                #                 第一次边界触发判断，意味着需要保存最后一次裁剪结果了
                if up + self.subsize >= height:
                    up = max(height - self.subsize, 0)
                #                 为了防止裁剪的subsize>输入图像的长/宽的情况
                right = min(left + self.subsize, weight - 1)
                down = min(up + self.subsize, height - 1)
                subimgname = outbasename + str(left) + '___' + str(up)
                # self.f_sub.write(name + ' ' + subimgname + ' ' + str(left) + ' ' + str(up) + '\n')

                #                 根据分割的坐标进行裁剪子区域并保存
                self.savepatches(resizeimg, objects, subimgname, left, up, right, down)

                #                 第二次触发判断，离开循环
                if up + self.subsize >= height:
                    break  # 只break这一层的while
                else:
                    up = up + self.slide

            #             先裁剪竖直方向，再进行水平方向平移
            if left + self.subsize >= weight:
                break
            else:
                left = left + self.slide

    #          切割图像集成模块，一次裁剪一个文件夹内的所有图片
    def splitdata(self, rate):
        """
        :param rate: resize rate before cut
        """
        imagelist = GetFileFromThisRootDir(self.imagepath)  # 获取图片名字列表
        imagenames = [util.custombasename(x) for x in imagelist if (util.custombasename(x) != 'Thumbs')]
        print(imagenames)
        for name in imagenames:
            self.SplitSingle(name, rate, self.ext)


if __name__ == '__main__':
    # example usage of ImgSplit
    split = splitbase(r'example',
                      r'examplesplit')
    split.splitdata(1)
