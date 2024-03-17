import random
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import random
from PIL import Image
import numpy as np
from torchvision import transforms
from itertools import islice, chain
import os
import cv2

def center_crop(image):
  center = image.shape[0] / 2, image.shape[1] / 2
  if center[1] < 256 or center[0] < 256:
    return cv2.resize(image, (256, 256))
  x = center[1] - 128
  y = center[0] - 128

  return image[int(y):int(y+256), int(x):int(x+256)]

class MyCustomDataset(Dataset):
    def __init__(self, 
                 path_gt,
                 device='cpu'
                ):
        
        self._items = [] 
        self._index = 0
        self.device = device
        dir_img = sorted(os.listdir(path_gt))
        img_pathes = dir_img

        for img_path in img_pathes:
          self._items.append((
            os.path.join(path_gt, img_path)
          ))
        random.shuffle(self._items)

    def __len__(self):
      return len(self._items)

    def next_data(self):
      gt_path = self._items[self._index]
      self._index += 1 
      if self._index == len(self._items):
        self._index = 0
        random.shuffle(self._items)

      image = Image.open(gt_path).convert('RGB')
      image = np.array(image).astype(np.float32) 
      image = center_crop(image)

      image = image / 255.
      image = transforms.ToTensor()(image)
      y = image.to(self.device)
      return y

    def __getitem__(self, index):
      gt_path = self._items[index]
      image = Image.open(gt_path).convert('RGB')
      image = np.array(image).astype(np.float32) 

      image = center_crop(image)

      image = image / 255.
      image = transforms.ToTensor()(image)
      y = image.to(self.device)
      return y
