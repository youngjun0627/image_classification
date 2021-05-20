import os
import csv

def train_validation_split(train_csv_path, validation_csv_path, split_ratio, train_path = './train'):
    train_csv = csv.writer(open(train_csv_path, 'w', encoding='utf-8-sig', newline=''))
    validation_csv = csv.writer(open(validation_csv_path, 'w', encoding='utf-8-sig', newline=''))
    index = 1
    #X_train = []
    #Y_train = []
    for label in os.listdir(train_path):
        image_root_path = os.path.join(train_path, label)
        for image_name in os.listdir(image_root_path):
            image_path = os.path.join(image_root_path, image_name)
            if index%(int(1/split_ratio))!=0:
                train_csv.writerow([image_path, label])
            else:
                validation_csv.writerow([image_path, label])
            index+=1

def test_csv(test_csv_path, test_path='./test'):
    test_csv = csv.writer(open(test_csv_path, 'w', encoding='utf-8-sig', newline=''))
    for label in os.listdir(test_path):
        image_root_path = os.path.join(test_path, label)
        for image_name in os.listdir(image_root_path):
            image_path = os.path.join(image_root_path, image_name)
            test_csv.writerow([image_path, label])

if __name__=='__main__':
    train_csv_path = './train.csv'
    validation_csv_path = './validation.csv'
    test_csv_path = './test.csv'
    split_ratio = 0.1
    
    train_validation_split(train_csv_path, validation_csv_path, split_ratio)
    
    test_csv(test_csv_path)



