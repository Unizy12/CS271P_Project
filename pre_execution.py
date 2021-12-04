import requests
import os
import time
import zipfile
import numpy as np
import scipy.io as sio
import cv2
import csv
import random

EGOHANDS_DATASET_URL = "http://vision.soic.indiana.edu/egohands_files/egohands_data.zip"
Target_Path = "egohands_data.zip"
Image_Dir = 'egohands/_LABELLED_SAMPLES'

def download_data(url, data_path):
    if not os.path.exists(data_path):
        print('start downloading egohands dataset.')
        req = requests.get(url, stream=True)
        with open(data_path, 'wb') as data_set:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    data_set.write(chunk)
        time.sleep(1)
        print('download finished')

def extract_zipfile(data_path):
    if not os.path.exists('egohands'):
        print('Start extracting files.')
        with zipfile.ZipFile(data_path, 'r') as zip_ref:
            zip_ref.extractall('egohands')
        print('Extraction finished.')

Image_Dir = 'egohands/_LABELLED_SAMPLES'
# def rename_files(image_dir):
#     for root, dirs, filesname in os.walk(image_dir):
#         for dir in dirs:
#             for f in os.listdir(image_dir +'/'+ dir):
#                 if f.split('.')[1] == 'jpg':
#                     os.rename(image_dir +'/'+ dir + '/' + f, image_dir + '/' + dir + '_' + f)
def save_csv(path, data):
    with open(path, 'w') as csvfile:
        wr = csv.writer(csvfile)
        for i in data:
            wr.writerow(i)

def mark_hand(image_dir):
    for root, dirs, filesname in os.walk(image_dir):
        for dir in dirs:
            record = []
            print('---It is dealing with fold %s---' %dir)
            for f in os.listdir(image_dir+'/'+dir):
                if f.split('.')[1] == 'jpg':
                    record.append(f)
            record.sort()
            execute_every_file(image_dir+'/'+dir, record)

def execute_every_file(file_father_path, record):
    rectangles = sio.loadmat(file_father_path+'/polygons.mat')
    polygons = rectangles['polygons'][0]
    count = 0
    for pointlists in polygons:
        img_name = record[count]
        font = cv2.FONT_HERSHEY_SIMPLEX
        img_path = file_father_path+'/'+img_name
        img = cv2.imread(img_path)

        count += 1

        experiment_data = []
        for pointlist in pointlists:
            if len(pointlist) <= 1:
                continue
            pst = np.empty((0,2),int)
            max_x = max_y = 0
            min_x = min_y = 2**32
            
            pidx = 0
            for point in pointlist:
                if len(point)==2:
                    x = int(point[0])
                    y = int(point[1])
                    max_x = x if (x > max_x) else max_x
                    min_x = x if (x < min_x) else min_x
                    max_y = y if (y > max_y) else max_y
                    min_y = y if (y < min_y) else min_y
                    appeno = np.array([[x,y]])
                    pst = np.append(pst,appeno, axis=0)
                    cv2.putText(img, ".", (x, y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            if min_x > 0 and min_y > 0 and max_x > 0 and max_y > 0:
                # imgname, class, img_width, img_height, min_x, min_y, max_x, max_y
                data_row = [img_name, 'hand', np.size(img, 1),np.size(img, 0), min_x, min_y, max_x, max_y]
                experiment_data.append(data_row)
            cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 1)
            cv2.polylines(img, [pst], True, (0, 255, 255), 1)

        csv_path = file_father_path
        if not os.path.exists(csv_path+'/'+img_name.split('.')[0]+'.csv'):
            cv2.putText(img, "DIR : " + file_father_path + " - " + img_name, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
            cv2.imshow('Verifying annotation ', img)
            save_csv(csv_path+'/'+img_name.split('.')[0] + ".csv", experiment_data)
            print("===== saving csv file for ", img_name)
        cv2.waitKey(2)  # close window when a key press is detected
def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
def data_spilit(Image_Dir):
    create_directory('images')
    create_directory('images/train')
    create_directory('images/test')

    data_size = 4000
    loop_idx = 0
    data_sampsize = int(0.1*data_size)
    test_samp_array = random.sample(range(data_size), k=data_sampsize)

    for root, dirs, filenames in os.walk(Image_Dir):
        for dir in dirs:
            for f in os.listdir(Image_Dir+'/'+dir):
                if f.split('.')[1] == 'jpg':
                    loop_idx += 1

                    if loop_idx in test_samp_array:
                        os.rename(Image_Dir+'/'+dir+'/'+f,'images/test/'+f)
                        os.rename(Image_Dir+'/'+dir+'/'+f.split('.')[0]+'.csv', 'images/test/'+f.split('.')[0]+'.csv')
                    else:
                        os.rename(Image_Dir+'/'+dir+'/'+f,'images/train/'+f)
                        os.rename(Image_Dir+'/'+dir+'/'+f.split('.')[0]+'.csv', 'images/train/'+f.split('.')[0]+'.csv')
            os.remove(Image_Dir+'/'+dir+'/polygons.mat')
            os.rmdir(Image_Dir+'/'+dir)

def generate_label_files(dir_path):
    header = ['filename', 'class', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax']
    for root, dirs, files in os.walk(dir_path):
        for dir in dirs:
            csvholder = []
            csvholder.append(header)
            loop_index = 0
            for f in os.listdir(dir_path+'/'+dir):
                if f.split('.')[1] == 'csv':
                    loop_index += 1
                    csv_file = open(dir_path+'/'+dir+'/'+f,'r')
                    reader = csv.reader(csv_file)
                    for row in reader:
                        csvholder.append(row)
                    csv_file.close()
                    os.remove(dir_path+'/'+dir+'/'+f)
            save_csv(dir_path+'/'+dir+'/'+dir+'_labels.csv', csvholder)
            print('Is saving %s'%dir_path+'/'+dir+'/'+dir+'_labels.csv')

if __name__ == '__main__':
    # download_data(EGOHANDS_DATASET_URL,Target_Path)
    extract_zipfile(Target_Path)
    mark_hand(Image_Dir)
    data_spilit(Image_Dir)
    generate_label_files("images/")