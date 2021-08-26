# The code is used for visulization, inspired from cocoapi
#  Licensed under the Simplified BSD License [see bsd.txt]

import os
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Circle
import numpy as np
import datasets.DOTA_devkit.dota_utils as util
from collections import defaultdict
import cv2


def _isArrayLike(obj):
    if type(obj) == str:
        return False
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


# 用来演示分割与合并用
class DOTA:
    def __init__(self, basepath):
        self.basepath = basepath  # 包含图片与标签的总文件夹的路径
        self.labelpath = os.path.join(basepath, 'labelTxt')  # 标签文件夹路径
        self.imagepath = os.path.join(basepath, 'images')  # 图片文件夹路径
        self.imgpaths = util.GetFileFromThisRootDir(self.labelpath)  # 获取所有标签文件的路径(1个标签文件对应1张图片)
        self.imglist = [util.custombasename(x) for x in self.imgpaths]  # 获取图片文件名列表(除后缀)
        self.catToImgs = defaultdict(list)  # value=list类型的默认字典_KEY:类别,VALUE:该类别的目标所在的图片名字
        self.ImgToAnns = defaultdict(list)  # value=list类型的默认字典_KEY:图片名字,VALUE:图片下所有目标信息
        self.createIndex()

    #     建立catToImgs/ImgToAnns的 索引
    def createIndex(self):
        for filename in self.imgpaths:
            objects = util.parse_dota_poly(filename)  # 获取一张图片中的目标数据,类型为列表
            imgid = util.custombasename(filename)  # 获取当前图像文件名字
            self.ImgToAnns[imgid] = objects  # 建立字典映射,一幅图对应一个object列表
            for obj in objects:
                cat = obj['name']
                self.catToImgs[cat].append(imgid)

    #     获取包含特定类别下的图片名字
    def getImgIds(self, catNms=[]):
        """
        :param catNms: 类别名字
        :return: 包含在输入类别名字中所有图片的ID
        """
        #         对输入类别的数据类型做判断,如果不是列表且不具有长度和迭代功能,将其变成列表
        catNms = catNms if _isArrayLike(catNms) else [catNms]

        #         如果没有输入,默认返回图片文件名列表(除后缀)
        if len(catNms) == 0:
            return self.imglist
        #         有输入
        else:
            imgids = []
            for i, cat in enumerate(catNms):
                if i == 0:
                    #                     统计当前类别所在的图片名字(无重复)
                    imgids = set(self.catToImgs[cat])
                else:
                    #             取出图片(该图片包含所有所有所需要的类别的物体)
                    imgids &= set(self.catToImgs[cat])
        #     最后将其list化,因为set不支持遍历
        return list(imgids)

    #     读取对应图像和类别标签
    def loadAnns(self, catNms=[], imgId=None, difficult=None):
        """
        :param catNms: category names
        :param imgId: the img to load anns
        :return: objects
        """
        #         列表化
        catNms = catNms if _isArrayLike(catNms) else [catNms]

        #         先读取这幅图像对应的总体标签
        objects = self.ImgToAnns[imgId]

        if len(catNms) == 0:
            return objects
        #         进一步指定标签中的类别
        outobjects = [obj for obj in objects if (obj['name'] in catNms)]
        return outobjects

    #     显示标注内容
    def showAnns(self, objects, imgId, range):
        """
        :param catNms: category names
        :param objects: objects to show
        :param imgId: img to show
        :param range: display range in the img
        :return:
        """
        #         显示图像
        img = self.loadImgs(imgId)[0]
        print('显示标注过程中原图尺寸：', img.shape)

        plt.imshow(img)
        plt.axis('off')

        ax = plt.gca()  # 获取坐标轴
        ax.set_autoscale_on(False)  # 设置是否对绘图命令自动缩放
        polygons = []
        color = []
        circles = []
        r = 3.5

        #         开始遍历标签内容
        for obj in objects:
            poly = obj['poly']
            polygons.append(Polygon(poly))  # 创建1个多边形对象

            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]  # 随机颜色绘图
            color.append(c)

            point = poly[0]  # 取出左上角坐标（x,y）
            circle = Circle((point[0], point[1]), r)  # 创建1个圆对象（以半径为r，左上角坐标为圆心画圆）
            circles.append(circle)
        #         覆盖标注框
        #         p = PatchCollection(polygons, facecolors=color, linewidths=0, alpha=0.4)
        #         ax.add_collection(p)
        #         画边框
        p = PatchCollection(polygons, facecolors='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)
        #         画左上角顶点
        p = PatchCollection(circles, facecolors='red')
        ax.add_collection(p)
        #         保存图片
        plt.savefig('AnnTest' + imgId + '.jpg', bbox_inches="tight", pad_inches=0.0)

    #         输入一张特定的图像名字
    def loadImgs(self, imgids=[]):
        """
        :param imgids: integer ids specifying img
        :return: loaded img objects
        """
        #         对输入list化
        print('loadImg_isArrayLike:', _isArrayLike(imgids))
        imgids = imgids if _isArrayLike(imgids) else [imgids]
        print('imgids_afterList:', imgids)
        #         遍历图像名字
        imgs = []
        for imgid in imgids:
            #             读取具体的图片
            filename = os.path.join(self.imagepath, imgid + '.png')
            print('filename:', filename)
            img = cv2.imread(filename)  # 使用opencv读取
            imgs.append(img)  # 取出图片数据保存在列表中
        return imgs

# if __name__ == '__main__':
#     examplesplit = DOTA('examplesplit')
#     imgids = examplesplit.getImgIds(catNms=['plane'])
#     img = examplesplit.loadImgs(imgids)
#     for imgid in imgids:
#         anns = examplesplit.loadAnns(imgId=imgid)
#         examplesplit.showAnns(anns, imgid, 2)
