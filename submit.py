import torch
from dataset import MyDataset
from model import MyModel
from activate import test
from transform import create_validation_transform
import os
from torch.utils.data import DataLoader

save_path = './latest.pth'
device = torch.device('cuda:0')
num_classes = 10
root = '.'

def get_classes():
    classes = set()

    train_path = '../train'
    test_path = '../test'

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
    return classes

classes = get_classes()
validation_transform = create_validation_transform(True)
test_dataset = MyDataset(transform = validation_transform, mode = 'test', classes = classes)
model = MyModel(num_classes = len(classes))
model.load_state_dict(torch.load(save_path))
model = model.to(device)
test_dataloader = DataLoader(test_dataset,\
        batch_size=2,\
        shuffle = False
        )

test(model, test_dataloader, None,  None, device)


