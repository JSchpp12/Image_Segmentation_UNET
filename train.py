from utils import convert_label_to_rgb 
import torchvision.transforms as transforms
from model import multi_unet_model
import tensorflow as tf
import keras
import os
import numpy as np
import pandas
# from dataset import PASCAL2007Dataset
from ohe import Ohe
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.AUTOTUNE

LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 50
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False

model = multi_unet_model(n_classes=22, IMG_HEIGHT=224, IMG_WIDTH=224, IMG_CHANNELS=3)
loss = keras.losses.CategoricalCrossentropy()
optimizer = keras.optimizers.AdamW(learning_rate = LEARNING_RATE)
metrics = ["accuracy"]


train_dir = os.path.join('data', 'pascal', 'train_JPEGImages')
mask_dir = os.path.join('data', 'pascal', 'train_SegClasses')
mask_val_dir = os.path.join('data', 'pascal', 'val_SegClasses')

val_dir = os.path.join('data', 'pascal', 'val_JPEGImages')

# mask_dir = 'data\\pascal\\train_SegClasses'
# val_dir = 'data\\pascal\\val_JPEGImages'
# mask_val_dir = 'data\\pascal\\val_SegClasses'
# transform = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),  transforms.ToTensor(), transforms.ToPILImage() ])

raw = pandas.read_csv('colormap.csv')
raw = raw.drop(['Class'], axis=1)
color_classes = tf.convert_to_tensor(raw, dtype=float)

def decode_image_jpg(path): 
    img = tf.io.decode_png(path, channels=3)
    
    return tf.image.resize(img, [224, 224])

def decode_image_png(path): 
    img = tf.io.decode_png(path, channels=3)
    
    return tf.image.resize(img, [224, 224])

def get_label_image_train(path): 
    # parts = tf.strings.split(path, os.path.sep)
    parts = tf.strings.split(path, 'train_JPEGImages')
    #get the file name since they are the same
    label_file = tf.strings.join([parts[0], tf.constant('train_SegClasses'), parts[1]])
    #label files are pngs need to remove .jpg and replace with png
    label_file = tf.strings.regex_replace(label_file, '.jpg', '.png')
    
    raw_label_file = tf.io.read_file(label_file)
    label_img = decode_image_png(raw_label_file)

    return label_img

def get_label_image_val(path): 
    # parts = tf.strings.split(path, os.path.sep)
    parts = tf.strings.split(path, 'val_JPEGImages')
    #get the file name since they are the same
    label_file = tf.strings.join([parts[0], tf.constant('val_SegClasses'), parts[1]])
    #label files are pngs need to remove .jpg and replace with png
    label_file = tf.strings.regex_replace(label_file, '.jpg', '.png')
    
    raw_label_file = tf.io.read_file(label_file)
    label_img = decode_image_png(raw_label_file)

    return label_img

def calculate_label_encoding(label_image): 
    image_reshape = tf.reshape(label_image, [-1, 3])
    image_reshape = tf.expand_dims(image_reshape, 1)
    # tf.print(image_reshape)
    
    diff = tf.subtract(image_reshape, color_classes)
    square_diff = tf.square(diff)
    
    
    # tf.print(square_diff)
    dists = tf.reduce_sum(square_diff, -1)
    # tf.print(dists)
    # print(dists)
    indicies = tf.argmin(dists, axis=-1)
    # print(indicies)

    # zero = tf.constant(0, dtype=tf.int64)
    # where = tf.not_equal(indicies, zero)
    # where = tf.where(where)
    # print(where)
    # tf.print(indicies[where[0][0]])
    
    indicies = tf.reshape(indicies, (224, 224))
    indicies = tf.one_hot(indicies, 22)
    return indicies

def process_path(path): 
    image = tf.io.read_file(path)
    image = decode_image_jpg(image)
    
    label_image = get_label_image_train(path)
    flatten_label_encode = calculate_label_encoding(label_image)
    
    return image, flatten_label_encode
    
def process_path_val(path): 
    image = tf.io.read_file(path)
    image = decode_image_jpg(image)
    
    label_image = get_label_image_val(path)
    flatten_label_encode = calculate_label_encoding(label_image)
    
    return image, flatten_label_encode

train_ds = tf.data.Dataset.list_files(f'{train_dir}/*.jpg')
train_ds = (train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
            .batch(16))

val_ds = tf.data.Dataset.list_files(f'{val_dir}/*.jpg')
val_ds = (val_ds.map(process_path_val, num_parallel_calls=AUTOTUNE)
            .batch(16))

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.getcwd(), histogram_freq=1)

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.fit(train_ds, validation_data=val_ds, epochs=NUM_EPOCHS, callbacks=[tensorboard_callback])
