import os
import csv
import numpy as np
import cv2
from torch.utils.data.dataset import Dataset
from transform import create_train_transform, create_validation_transform
import torch

class MyDataset(Dataset):
    def __init__(self, root='.', transform = None, mode='train', classes=None):
        super(MyDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.mode = mode
        if classes is None:
            raise Exception('needed classes')
        self.labels = []
        self.images = []
        self.index_labels = {cls:0 for cls in classes}
        for i, cls in enumerate(classes):
            self.index_labels[cls]=i
            data_path = '{}.csv'.format(mode)

        with open(data_path, 'r', encoding='utf-8-sig') as f:
            rdr = csv.reader(f)
            for line in rdr:
                image_path = line[0]
                label = line[1]
                self.images.append(image_path)
                self.labels.append(label)
    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]
        img = self.read_image(img)
        label = self.convert_label(label)
        if self.transform:
            img = self.transform(image = img)['image']
            return img, label
    def __len__(self):
        return len(self.images)

    def read_image(self, img):
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def convert_label(self, label):
        return self.index_labels[label]

    def get_class_weights(self):
        weights = [0 for _ in range(len(self.index_labels))]
        for label in self.labels:
            weights[self.index_labels[label]]+=1
        weights = np.array(weights)
        weights = weights / weights.sum()
        weights = 1.0 / weights
        weights = weights / weights.sum()
        return weights

    def get_class_weights2(self):
        weights = [0 for _ in range(len(self.index_labels))]
        for label in self.labels:
            weights[self.index_labels[label]]+=1
        weights = np.array(weights)
        normedWeights = [1 - (x / sum(weights)) for x in weights]
        return normedWeights


if __name__=='__main__':

    classes = set()

    train_path = './train'
    test_path = './test'

    total_train_num = 0
    total_test_num = 0
    for label in os.listdir(train_path):
        classes.add(label)
        image_num = len(os.listdir(os.path.join(train_path,label)))
        total_train_num += image_num
        print('train dataset size : {} -> {}'.format(label,image_num))

    for label in os.listdir(test_path):
        image_num = len(os.listdir(os.path.join(test_path,label)))
        total_test_num += image_num
        print('test dataset size : {} -> {}'.format(label,image_num))
        print('total train dataset : {} \t total test dataset : {}'.format(total_train_num, total_test_num))      


    train_transform = create_train_transform(True, True, True, True)
    train_dataset = MyDataset(transform = train_transform, classes = classes)
    val_transform = create_validation_transform(True)
    val_dataset = MyDataset(transform = val_transform, mode = 'validation', classes = classes)
    test_dataset = MyDataset(transform = val_transform, mode = 'test', classes = classes)
    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))
    print(train_dataset.get_class_weights())
    print(train_dataset.get_class_weights2())
