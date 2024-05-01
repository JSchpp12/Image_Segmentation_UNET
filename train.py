from utils import convert_label_to_rgb 
import torchvision.transforms as transforms
from model import multi_unet_model
import torch.optim as optim
import tensorflow as tf
import keras
import os
import numpy as np
from dataset import PASCAL2007Dataset
from ohe import Ohe
import matplotlib.pyplot as plt

LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False


model = multi_unet_model(n_classes=22, IMG_HEIGHT=224, IMG_WIDTH=224, IMG_CHANNELS=3)
loss = keras.losses.CategoricalCrossentropy()
optimizer = keras.optimizers.AdamW(learning_rate = LEARNING_RATE)
metrics = ["accuracy"]


train_dir = 'data\\train_JPEGImages'
mask_dir = 'data\\train_SegClasses'
val_dir = 'data\\val_JPEGImages'
mask_val_dir = 'data\\val_SegClasses'
transform = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),  transforms.ToTensor(), transforms.ToPILImage() ])


train_indices = os.listdir(train_dir)
train_indices = list(map(lambda x: x[:x.index('.')], train_indices))

val_indices = os.listdir(val_dir)
val_indices = list(map(lambda x: x[:x.index('.')], val_indices))


x = PASCAL2007Dataset(image_dir=train_dir, mask_dir=mask_dir, transform=transform)

images = []
masks = []
for f in train_indices[:16]:
    img_mask = x[f]
    images.append(np.array(img_mask['image']))
    mask = np.array(img_mask['mask'])
    y = Ohe(mask, 22)
    mask_ohe = y.one_hot_encoded_mask()
    masks.append(mask_ohe)

images = np.array(images)
masks = np.array(masks)

#print(images.shape, masks.shape)

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.fit(images, masks, epochs=5, batch_size=4, verbose=2)

y = PASCAL2007Dataset(image_dir=val_dir, mask_dir=mask_val_dir, transform=transform)
val_images = []
val_masks = []
for f in val_indices[:16]:
    img_mask = y[f]
    val_images.append(np.array(img_mask['image']))
    mask = np.array(img_mask['mask'])
    y1 = Ohe(mask, 22)
    mask_ohe = y1.one_hot_encoded_mask()
    val_masks.append(mask_ohe)

val_images = np.array(val_images)
val_masks = np.array(val_masks)

model.evaluate(val_images, val_masks, batch_size=4, verbose=2)


"""
TO VISUALIZE PREDICTED IMAGES

test = x['000032']
test_img = np.array(test['image'])
test_img = test_img[None, :, :, :]
b = model.predict(test_img)
pred_b = np.argmax(b, axis=3)[0, :, :]
img = convert_label_to_rgb(pred_b)

plt.imshow(img)
plt.show()
"""
