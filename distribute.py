import pandas as pd
import os
import shutil

file_path = 'metadata.xlsx'
xlsx = pd.read_excel(file_path)
labels = xlsx['primary_microconstituent']
graphs = xlsx['path']
ori_directory = 'original_dataset'
ori_path = ''
for i in range(598):
    if not os.path.exists(labels[i]):
        os.makedirs(labels[i])
    ori_path = ori_directory + '/Cropped' + graphs[i]

    shutil.move(ori_path, labels[i])
