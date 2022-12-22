import glob
import os
from sympy import Nor
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2

class PhoneNumberDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.phone_numbers_frame = pd.read_csv(csv_file)
        self.trasform = transform
    def __len__(self):
        return len(self.phone_numbers_frame)
    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, self.phone_numbers_frame.iloc[idx, 0])
        
        image = cv2.imread(image_name)
        if(image.shape[2] == 4):
            image = image[:, :, 0:3]
        
        phone_number = str(self.phone_numbers_frame.iloc[idx, 1])
        if(len(phone_number) != 10):
            s = '0' * (10 - len(phone_number))
            phone_number = s + phone_number
        sample = {'image' : image.copy(), 'phone_number' : phone_number}

        if self.trasform:
            sample = self.trasform(sample)

        return sample

class Gray_scale_Image(object):
    def __call__(self, sample):
        if isinstance(sample, dict):
            image, phone_number = sample['image'], sample['phone_number']
            image_copy = np.copy(image)
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
            return {'image': image_copy, 'phone_number': phone_number}
        else:
            image = sample
            image_copy = np.copy(image)
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
            return image_copy

class Normalize(object):
    def __call__(self, sample):
        if isinstance(sample, dict):
            image, phone_number = sample['image'], sample['phone_number']
            image_copy = np.copy(image)
            image_copy = image_copy / 255.0
            return {'image': image_copy, 'phone_number': phone_number}
        else:
            image_copy = np.copy(sample)
            image_copy = image_copy / 255.0
            return image_copy

class Gray_scale_to_Binary(object):
    def __call__(self, sample):
        if isinstance(sample, dict):
            image, phone_number = sample['image'], sample['phone_number']
            image_copy = np.copy(image)
            _, image_copy = cv2.threshold(image_copy, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            return {'image': image_copy, 'phone_number': phone_number}
        else:
            image_copy = np.copy(sample)
            _, image_copy = cv2.threshold(image_copy, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            return image_copy

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, (new_w, new_h))
        return img

class Dilation(object):
    def __call__(self, image):
        image_copy = np.copy(image)
        kernel = np.ones((2,2), np.uint8)
        image_copy = cv2.dilate(image_copy, kernel, iterations=1)
        return image_copy

class MedianFilter(object):
    def __call__(self, sample):
        if isinstance(sample, dict):
            image, phone_number = sample['image'], sample['phone_number']
            image_copy = np.copy(image)
            image_copy = cv2.medianBlur(image_copy, 3)
            return {'image': image_copy, 'phone_number': phone_number}
        else:
            image_copy = np.copy(sample)
            image_copy = cv2.medianBlur(image_copy, 3)
            return image_copy

class Erosion(object):
    def __call__(self, image):
        image_copy = np.copy(image)
        kernel = np.ones((2,2), np.uint8)
        image_copy = cv2.erode(image_copy, kernel, iterations=1)
        return image_copy

class Inversion(object):
    def __call__(self, image):
        image_copy = np.copy(image)
        image_copy = 255 - image_copy
        return image_copy

class BoundingBoxes(object):
    def __call__(self, sample):

        if isinstance(sample, dict):
            image, phone_number = sample['image'], sample['phone_number']
        else:
            image = sample

        rescale = Rescale((28, 28))
        image_copy = np.copy(image)
        contours, heirarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contours_poly = [None] * len(contours)
        bound_rect = []

        for i, c in enumerate(contours):
            if heirarchy[0][i][3] == -1:
                contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                bound_rect.append(cv2.boundingRect(contours_poly[i]))

        bound_rect.sort(key=lambda x: -x[2] * x[3])
        bound_rect = bound_rect[0:10]
        bound_rect.sort(key=lambda x : x[0])

        image_list_digits = []
        digits_list = []

        i = 0
        for box in bound_rect:
            x, y, h, w = box
            if(x > 1 and y > 1):
                imgTemp = rescale(image_copy[int(y - 1) : int(y + w + 1), int(x - 1) : int(x + h + 1)])
            elif(x > 2 and y > 2):
                imgTemp = rescale(image_copy[int(y - 2) : int(y + w + 2), int(x - 2) : int(x + h + 2)])
            else:
                imgTemp = rescale(image_copy[int(y) : int(y + w), int(x) : int(x + h)])
            image_list_digits.append(imgTemp)

            if isinstance(sample, dict):
                digits_list.append(int(phone_number[i]))
                i+=1
            

        for i in range(10):
            sample_Temp = image_list_digits[i]
            for j, fx in enumerate([MedianFilter(),Dilation(),Erosion()]):
                transformed_img = fx(sample_Temp)
                sample_Temp = transformed_img
            image_list_digits[i] = sample_Temp
                
        if isinstance(sample, dict):
            return {'digit_Images_list': image_list_digits, 'digits_list': digits_list}
        else:
            return image_list_digits

class ToTensor(object):
    def __call__(self, sample):
        if isinstance(sample, dict):
            digit_Images_list, digits_list = sample['digit_Images_list'], sample['digits_list']
        else:
            digit_Images_list = sample
        image_copy = np.copy(digit_Images_list)
        # image_copy = image_copy.transpose((2, 0, 1))
        if isinstance(sample, dict):
            return{'image': torch.from_numpy(image_copy), 'digits_list': digits_list}
        else:
            return torch.from_numpy(image_copy)