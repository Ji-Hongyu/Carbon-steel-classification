import cv2
import cv2 as cv
import numpy
import numpy as np
import pandas as pd
import os
import shutil
import torch
from torch.utils.data import Dataset


def distribute():
    file_path = './data/metadata.xlsx'
    xlsx = pd.read_excel(file_path)
    labels = xlsx['primary_microconstituent']
    graphs = xlsx['path']
    ori_directory = 'original_dataset'
    for i in range(598):
        if not os.path.exists(labels[i]):
            os.makedirs(labels[i])
        ori_path = ori_directory + '/Cropped' + graphs[i]
        shutil.move(ori_path, labels[i])


def cut():
    classes = ['pearlite+spheroidite', 'martensite', 'network', 'pearlite', 'spheroidite', 'widmanstatten']
    for label in classes:
        if not os.path.exists("./data_divided_into6/" + label):
            os.makedirs("./data_divided_into6/" + label)
        for counter in range(1, 2000):
            img_path = "./data/" + label + "/Croppedmicrograph" + str(counter)+".png"
            if os.path.exists(img_path):
                img = cv.imread(img_path)
            else:
                continue
            h, w, c = img.shape
            new_img1 = img[0:int(h/2), 0:int(w/3)]
            new_img2 = img[0:int(h/2), int(w/3):int(w*2/3)]
            new_img3 = img[0:int(h/2), int(w*2/3):w]
            new_img4 = img[int(h/2):h, 0:int(w/3)]
            new_img5 = img[int(h/2):h, int(w/3):int(w*2/3)]
            new_img6 = img[int(h/2):h, int(w*2/3):w]
            cv.imwrite("./data_divided_into6/" + label + "/Croppedmicrograph"+str(counter)+"(1).png", new_img1)
            cv.imwrite("./data_divided_into6/" + label + "/Croppedmicrograph"+str(counter)+"(2).png", new_img2)
            cv.imwrite("./data_divided_into6/" + label + "/Croppedmicrograph"+str(counter)+"(3).png", new_img3)
            cv.imwrite("./data_divided_into6/" + label + "/Croppedmicrograph"+str(counter)+"(4).png", new_img4)
            cv.imwrite("./data_divided_into6/" + label + "/Croppedmicrograph"+str(counter)+"(5).png", new_img5)
            cv.imwrite("./data_divided_into6/" + label + "/Croppedmicrograph"+str(counter)+"(6).png", new_img6)


def dataset_distribute():

    """shuffle and distribute the 4× enlarged dataset into
     train set, validation set and test set with the
     proportion of 8 : 1 : 1"""

    dataset_path = "data_divided_into6"
    if not os.path.exists(dataset_path + "/train"):
        os.makedirs(dataset_path + "/train")
    if not os.path.exists(dataset_path + "/val"):
        os.makedirs(dataset_path + "/val")
    if not os.path.exists(dataset_path + "/test"):
        os.makedirs(dataset_path + "/test")
    classes = ["martensite", "network", "pearlite", "pearlite+spheroidite", "spheroidite", "widmanstatten"]

    for folder in classes:
        graphs = np.array([dataset_path + "/" + folder + "/"
                           + file for file in os.listdir(dataset_path + "/" + folder)])
        np.random.get_state()
        np.random.shuffle(graphs)

        if not os.path.exists("data_divided_into6/train/" + folder):
            os.makedirs("data_divided_into6/train/" + folder)
        if not os.path.exists("data_divided_into6/val/" + folder):
            os.makedirs("data_divided_into6/val/" + folder)
        if not os.path.exists("data_divided_into6/test/" + folder):
            os.makedirs("data_divided_into6/test/" + folder)

        for i in range(len(graphs)):
            if i < int(len(graphs) * 0.8):
                shutil.move(graphs[i], "data_divided_into6/train/" + folder)
            if int(len(graphs) * 0.8) <= i < int(len(graphs) * 0.9):
                shutil.move(graphs[i], "data_divided_into6/val/" + folder)
            if int(len(graphs) * 0.9) <= i:
                shutil.move(graphs[i], "data_divided_into6/test/" + folder)


class MyDataset(Dataset):
    """overwrite the Dataset class to fit our prepocessed dataset to dataloader"""
    def __init__(self, file_path):
        super(MyDataset, self).__init__()
        dataset_lst = []
        label_lst = []
        target = 0
        for graph_set in os.listdir(file_path):
            for graph in os.listdir(file_path + "/" + graph_set):
                img = cv2.imread(file_path + "/" + graph_set + "/" + graph)
                img = cv2.resize(img, (224, 224))
                dataset_lst.append(img)
                label_lst.append(target)
            target += 1
        self.len = len(dataset_lst)
        self.dataset = torch.tensor(numpy.array(dataset_lst))
        self.labels = torch.tensor(numpy.array(label_lst))

    def __getitem__(self, index):
        return self.dataset[index], self.labels[index]

    def __len__(self):
        return self.len
