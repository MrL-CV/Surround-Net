from datasets.DOTA_devkit import ImgSplit
from datasets.DOTA_devkit import SplitOnlyImage
import os

# 训练输入文件夹
train_base_file_path = '/home/batfly_fluid01/桌面/SurroundNet/Project/data/DOTA/basepath_train'
# 训练输出文件夹
train_split_total_outpath = '/home/batfly_fluid01/桌面/SurroundNet/Project/data/DOTA/split_total_outpath'

# 测试输入文件夹
test_base_file_path = '/home/batfly_fluid01/桌面/SurroundNet/Project/data/DOTA/basepath_test'
# 测试输出文件夹
test_split_total_outpath = '/home/batfly_fluid01/桌面/SurroundNet/Project/data/DOTA/test_split_total_outpath'

# 训练输出文件夹中的图片名字获取文件夹
train_Image_name = os.path.join(train_split_total_outpath, 'images')
# 测试输出文件夹中的标签文件夹
test_Image_name = os.path.join(test_split_total_outpath)

# 汇总.txt产生的地址
Image_name_txt_dir = '/home/batfly_fluid01/桌面/SurroundNet/Project/data/DOTA'


def splitdata_1_05_img_and_label(a):
    if a:
        split_total = ImgSplit.splitbase(
            train_base_file_path,
            train_split_total_outpath,
            gap=100,
            subsize=600,
        )
    else:
        split_total = SplitOnlyImage.splitbase(
            test_base_file_path,
            test_split_total_outpath,
            gap=100,
            subsize=600
        )
    split_total.splitdata(1)
    split_total.splitdata(0.5)


def GetTheName(fileName, a):
    if a:
        outdir = os.path.join(Image_name_txt_dir, 'trainval.txt')
    else:
        outdir = os.path.join(Image_name_txt_dir, 'test.txt')
    with open(outdir, 'w') as f:
        for root, dirs, files in os.walk(fileName):
            # print(files)
            for file in sorted(files):
                strName = os.path.splitext(file)[0]
                f.write(strName + '\n')


if __name__ == '__main__':
    splitdata_1_05_img_and_label(False)
    GetTheName(test_Image_name, False)
