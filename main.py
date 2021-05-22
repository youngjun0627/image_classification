import torch
from dataset import MyDataset
from model import MyModel
from loss import MyLoss1, MyLoss2
from transform import create_train_transform, create_validation_transform
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from activate import train, validation

save_path = '../latest.pth'
device = torch.device('cuda:0')
num_classes = 10
root = '.'
BATCHSIZE = 64
LR = 0.01
EPOCHS = 10

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
train_transform = create_train_transform(True, True, True, True)
train_dataset = MyDataset(transform = train_transform, classes = classes)
validation_transform = create_validation_transform(True)
validation_dataset = MyDataset(transform = validation_transform, mode = 'validation', classes = classes)
model = MyModel(num_classes = len(classes)).to(device)
criterion = MyLoss1(weights = torch.tensor(train_dataset.get_class_weights(), dtype=torch.float32).to(device))
#criterion = MyLoss2(weights = torch.tensor(train_dataset.get_class_weights2(), dtype=torch.float32).to(device))
optimizer = optim.SGD(model.parameters(), lr = LR)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor = 0.5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15,20,25,30], gamma=0.1)
train_dataloader = DataLoader(train_dataset,\
        batch_size = BATCHSIZE,\
        shuffle = True,
        num_workers=4
        )
validation_dataloader = DataLoader(validation_dataset,\
        batch_size = BATCHSIZE,\
        shuffle = False
        )

pre_score = 0

for epoch in range(EPOCHS):

    train(model, train_dataloader, criterion, optimizer, device)
    if (epoch+1)%5==0:
        score = validation(model, validation_dataloader, criterion,  None, device)
        if score>pre_score:
            pre_score = score
            model = model.cpu()
            torch.save(model.state_dict(), save_path)
            model = model.to(device)
        scheduler.step(score)


