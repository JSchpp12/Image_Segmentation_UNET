import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms



# class PASCAL2007Dataset(Dataset):
#     def __init__(self, image_dir, mask_dir, transform=None):
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.transform = transform
#         self.images = os.listdir(image_dir)

#     def __len__(self):
#         return len(self.images)
    
#     def __getitem__(self, filename):
#         img_path = os.path.join(self.image_dir, filename + '.jpg')
#         mask_path = os.path.join(self.mask_dir, filename + '.png')
#         image = Image.open(img_path).convert("RGB")
#         mask = Image.open(mask_path).convert("RGB")
        
#         image = self.transform(image)
#         mask = self.transform(mask)
#         return {'image': image, 'mask': mask}
        

if __name__ == '__main__':
    image_dir = 'data\\train_JPEGImages\\'
    mask_dir = 'data\\train_SegClasses\\'
    transform = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST), transforms.ToTensor()])

    x = PASCAL2007Dataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
                                                                                            
    img_mask = x['000032']
    y = img_mask.shape