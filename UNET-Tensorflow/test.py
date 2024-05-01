import torchvision.transforms as transforms
import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image_dir = 'data\\JPEGImages\\'
mask_dir = 'data\\SegmentationClass'

i = cv2.imread(os.path.join(image_dir, '000032.jpg'))
j = Image.open(os.path.join(image_dir, '000032.jpg')).convert("RGB")
#print(np.array(j).shape)


transform = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST), transforms.ToTensor(), transforms.ToPILImage()])
j = transform(j)
print(np.array(j).shape)

plt.imshow(np.array(j))
plt.show()

# cv2.imshow('title', j)
# cv2.waitKey()
