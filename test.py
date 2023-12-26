import torchvision.models
from carbon_steel_classification.utils import *
from torch.utils.data import DataLoader
from torch import nn
import torch
import torchvision.models
from openvino.inference_engine import IECore
import time


def test():
    # 准备测试数据集
    test_path = './data_divided_into6/test'
    test_dataset = MyDataset(test_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    test_dataset_size = len(test_dataloader)

    # 加载模型
    model_path = './saved/6_Nopre/6_Nopre_vgg16_state_9.pth'
    vgg16 = torchvision.models.vgg16()
    vgg16.classifier[6] = nn.Sequential(nn.Linear(4096, 6))
    vgg16.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)

    # 测试开始
    total_accuracy = 0
    inf_time = time.time()
    for data in test_dataloader:
        imgs, targets = data
        imgs = imgs.to(torch.float32)
        targets = targets.to(torch.long)
        imgs = torch.reshape(imgs, (1, 3, 224, 224))

        outputs = vgg16(imgs)
        total_accuracy += (outputs.argmax(1) == targets).sum()
    inf_time = time.time() - inf_time

    # 测试结果
    print('模型 {} 在整体测试集上的准确率: {}'.format(model_path, total_accuracy / test_dataset_size))
    print('推理用时: {}'.format(inf_time))


def openvino_test():
    # 准备测试数据集
    test_path = './data_divided_into4/test'
    test_dataset = MyDataset(test_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    test_dataset_size = len(test_dataloader)

    # 加载模型
    model_path = './saved/4_Nopre'
    model_name = '4_Nopre_vgg16_state_8.pth'
    model = torchvision.models.vgg16()
    model.classifier[6] = nn.Sequential(nn.Linear(4096, 6))
    model.load_state_dict(torch.load(model_path + '/' + model_name, map_location='cpu'), strict=False)

    # 转换模型至onnx格式
    print(type(model))
    torch.onnx.export(model, (1, 3, 224, 224), "./onnx/4_Nopre_vgg16_state_8.onnx")

    # 转换onnx至IR
    # ------In root directory of openvino------
    # ------cmd------
    # python mo --input_model <model name> --output_dir <output dir> --input_shape [1, 3, 224, 224] --data_type FP16

    model_xml_path = './ir/4_pre_vgg16_state_7.xml'
    model_bin_path = './ir/4_pre_vgg16_state_7.bin'

    # 推理设置
    device = 'CPU'
    ie = IECore()
    net = ie.read_network(model=model_xml_path, weights=model_bin_path)
    exec_net = ie.load_network(network=net, device_name=device)

    # 开始推理
    total_accuracy = 0
    inf_time = time.time()
    for data in test_dataloader:
        img, target = data

        res = exec_net.infer(img)
        total_accuracy += (res.argmax(1) == target)
    inf_time = time.time() - inf_time

    # 测试结果
    print('模型 {} 在整体测试集上的准确率: {}'.format(model_name, total_accuracy / test_dataset_size))
    print('推理用时: {}'.format(inf_time))


test()
# openvino_test()
