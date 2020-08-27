import torch
import json
import sys
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

class ImageDataset(Dataset):


    def __init__(self, data_file, targets, feature_map_size=96):
        self.dataset = []
        with open(data_file) as fi:
            self.dataset = json.load(fi)
        self.feature_map_size = feature_map_size
        self.sampleN = len(self.dataset)
        self.targets = targets


    def __len__(self):
        return self.sampleN


    def __getitem__(self, idx):
        img_file = self.dataset[idx][0]
        label = self.dataset[idx][1]
        input_data = self.load_image(img_file)
        target = self.target_index(label)
        return input_data, target


    def load_image(self, file):
        with Image.open(file) as img:
            if img.height != self.feature_map_size:
                img = img.resize(((int(img.width*(self.feature_map_size/img.height)), self.feature_map_size)))
            img = img.convert('L')
            img = img.point(lambda x: 0 if x < 180 else 255)
            img = ImageOps.invert(img)
        data = np.asarray(img)
        if data.shape[1] < self.feature_map_size:
            data = np.pad(data,[(0,0),(0, self.feature_map_size-data.shape[1])], "constant")
        if data.dtype == np.uint8:
            data = (data/255).astype(np.float32)
        return torch.FloatTensor(data)


    def target_index(self, label):
        output = []
        for char in list(label):
            if char in self.targets:
                output.append(self.targets.index(char))
            else:
                print('{} is not target'.format(char) ,file=sys.stderr)
        return output


class ImageStrideDataset(ImageDataset):


    def __init__(self, data_file, targets, feature_map_size=(96, 96), stride=48):
        super(ImageStrideDataset, self).__init__(data_file, targets, feature_map_size=feature_map_size[0])
        self.feature_map_size = feature_map_size
        self.stride = stride


    def load_image(self, file):
        with Image.open(file) as img:
            #print(img.height, img.width)
            if img.height != self.feature_map_size[0]:
                img = img.resize(((int(img.width*(self.feature_map_size[0]/img.height)), self.feature_map_size[0])))
            img = img.convert('L')
            img = ImageOps.invert(img)
        data = np.asarray(img)
        data = np.pad(data,[(0,0),(self.stride, self.stride)], "constant")
        if data.dtype == np.uint8:
            data = (data/255).astype(np.float32)
        feature_map = np.empty((0, self.feature_map_size[0], self.feature_map_size[1]), np.float32)
        for index in range(0, data.shape[1], self.stride):
            if index+self.feature_map_size[1] > data.shape[1]:
                remind = data.shape[1] - index
                added_data = np.zeros(self.feature_map_size, dtype=np.float32)
                added_data[:, :remind] = data[:, index:]
                feature_map = np.append(feature_map, added_data[np.newaxis, :, :], axis=0)
                break
            else:
                feature_map = np.append(feature_map, data[np.newaxis, :, index:index+self.feature_map_size[1]], axis=0)
        return torch.FloatTensor(feature_map)