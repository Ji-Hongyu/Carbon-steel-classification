from utils import *


'''------------Data preprocess------------'''

# 利用元数据excel文档中给出label的对原始数据集的一部分进行划分
# distribute()

# 第一次划分后的数据集结合人工标注的数据集得到全部标注的数据集[各种类别的金相图片存放在以对应类别命名的目录中]
# 并对最终得到的完全标注数据集进行四等分分割扩充
# cut()

# 将标注、扩充后的数据集按比例 8:1:1 分为训练集(train目录)、验证集(val目录)和测试集(test目录)
# dataset_distribute()
