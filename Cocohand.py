import os
import random
import PIL.Image as Image
import csv
import cv2
Image_Dir = 'COCO-Hand/COCO-Hand-S/COCO-Hand-S_Images'
def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def data_spilit():
    create_directory('COCO-Hand/train')
    create_directory('COCO-Hand/test')

    data_size = 4535
    loop_idx = 0
    data_sampsize = int(0.1*data_size)
    test_samp_array = random.sample(range(data_size), k=data_sampsize)

    test_holder = []
    for root, dirs, filenames in os.walk(Image_Dir):
        for idx in range(len(filenames)):
            print('Is dealing with %s'%filenames[idx])
            if idx in test_samp_array:
                os.rename(Image_Dir+'/'+filenames[idx],'COCO-Hand/test/'+'COCOhand_'+filenames[idx])
            else:
                os.rename(Image_Dir+'/'+filenames[idx],'COCO-Hand/train/'+'COCOhand_'+filenames[idx])

def get_size(path):
    image = Image.open(path)
    width, height = image.size
    return [width, height]

def split_labeling():
    train_path = '/home/unizy/DeepLearningExamples/PyTorch/Detection/SSD/COCO-Hand/train'
    test_path = '/home/unizy/DeepLearningExamples/PyTorch/Detection/SSD/COCO-Hand/test'

    train = {}
    test = {}
    for root, dirs, filenames in os.walk(train_path):
        for filename in filenames:
            train[filename] = get_size(train_path+'/'+filename)
    for root, dirs, filenames in os.walk(test_path):
        for filename in filenames:
            test[filename] = get_size(test_path+'/'+filename)
    header = ['filename', 'class', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax']
    train_holder = []
    train_holder.append(header)
    test_holder = []
    test_holder.append(header)
    with open ('COCO-Hand/COCO-Hand-S/COCO-Hand-S_annotations.txt') as f:
        lines = f.readlines()
        for line in lines:
            image_name, xmin, xmax, ymin, ymax, x1, y1, x2, y2, x3, y3, x4, y4, category = line.strip('\n').split(',')
            image_name = 'COCOhand_' + image_name
            if image_name in train:
                width, height = train[image_name]
                row = [image_name, category, width, height, xmin, ymin,xmax, ymax]
                train_holder.append(row)
            else:
                width, height = test[image_name]
                row = [image_name, category, width, height, xmin, ymin,xmax, ymax]
                test_holder.append(row)
    
    save_csv('COCO-Hand/train.csv', train_holder)
    save_csv('COCO-Hand/test.csv', test_holder)

def save_csv(path, data):
    with open(path, 'w') as csvfile:
        wr = csv.writer(csvfile)
        for i in data:
            wr.writerow(i)

train_csv = 'COCO-Hand/train.csv'
test_csv = 'COCO-Hand/test.csv'
def check_pic(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] != 'filename':
                image = cv2.imread('COCO-Hand/train/'+row[0])
                start_point = (int(row[4]),int(row[5]))
                end_point = (int(row[6]),int(row[7]))
                image = cv2.rectangle(image, start_point, end_point,(255,0,0),2)
                cv2.imshow('Verifying annotation ', image)
                cv2.waitKey(100)  # close window when a key press is detected

if __name__ == '__main__':
    # data_spilit()
    # split_labeling()
    check_pic(train_csv)