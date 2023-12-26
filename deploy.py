import cv2
import torch
from torch import nn
import torchvision.models
import os


# 载入图片
# img_path = input('>>Imgpath:')
demoset = os.listdir('./data/demoset')
imgs = []
for img_path in demoset:
    img = cv2.imread('./data/demoset/' + img_path)
    img = cv2.resize(img, (224, 224))
    img = torch.tensor(img)
    img = torch.reshape(img, (1, 3, 224, 224))
    img = img.to(torch.float32)
    imgs.append(img)

# 加载模型
vgg16 = torchvision.models.vgg16()
vgg16.classifier[6] = nn.Sequential(nn.Linear(4096, 6))
vgg16.load_state_dict(torch.load('./saved/4_pre/4_pre_vgg16_state_7.pth', map_location='cpu'), strict=False)

# 输出分类结果
classes = ['马氏体', '网状结构', '珠光体', '珠光体+球状体', '球状体', '魏氏体']
info = ["高强度高硬度",
        "钢材内部缺陷之一, 对钢材力学性能很不利, 连成网状的碳化物使钢质变脆，韧性变坏",
        "塑性、韧性较好, 强度较高、硬度适中",
        "性能介于珠光体和球状珠光体之间",
        "珠光体球化后, 屈服点、抗拉强度、冲击韧性、蠕变极限和持久极限下降",
        "魏氏组织使钢的力学性能下降，尤其降低冲击性能"]
softmax = nn.Softmax(1)
vgg16.eval()
with torch.no_grad():
    for img in imgs:
        res = softmax(vgg16(img))
        idx = res.argmax(1)
        res = res.detach().numpy().tolist()
        print('图片中的金相组织可能为: {} | 概率: {}'.format(classes[idx], res[0][idx]))
        print('力学性能: ' + info[idx])
