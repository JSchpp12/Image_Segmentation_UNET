import shutil
import os

train_indices = 'data\\trainval.txt'
with open(train_indices) as f:
    jpeg_indices = []
    line = f.readline().strip()
    while line != '':
        jpeg_indices.append(line)
        line = f.readline().strip()

for i in jpeg_indices:
    shutil.move(os.path.join('data', 'JPEGImages', i + '.jpg'), 'data\\val_JPEGImages')
    shutil.move(os.path.join('data', 'SegmentationClass', i + '.png'), 'data\\val_SegClasses')

#shutil.move(os.path.join('data', 'JPEGImages', jpeg_indices[1] + '.jpg'), 'data\\train_JPEGImages')
#shutil.move(os.path.join('data', 'JPEGImages', jpeg_indices[2] + '.jpg'), 'data\\train_JPEGImages')

#print(jpeg_indices)


