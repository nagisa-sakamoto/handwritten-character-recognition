import torch
import json
import sys
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

class ImageDataset(Dataset):
    def __init__(self, data_file, targets, feature_map_size=(96, 96), stride=48):
        self.dataset = []
        with open(data_file) as fi:
            self.dataset = json.load(fi)

        self.sampleN = len(self.dataset)
        self.targets = targets
        self.feature_map_size = feature_map_size
        self.stride = stride


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


    def target_index(self, label):
        output = []
        for char in list(label):
            if char in self.targets:
                output.append(self.targets.index(char))
            else:
                print('{} is not target'.format(char) ,file=sys.stderr)
        return output


def _collate_fn(batch):
    def func(p):
        return p[0].size(0)

    batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)
    longest_sample = max(batch, key=func)[0]
    minibatch_size = len(batch)
    inputs = torch.zeros(minibatch_size, longest_sample.size(0), longest_sample.size(1), longest_sample.size(2))
    input_sizes = torch.IntTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        feature = sample[0]
        target = sample[1]
        seq_length = feature.size(0)
        inputs[x].narrow(0, 0, seq_length).copy_(feature)
        input_sizes[x] = seq_length
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_sizes, target_sizes


class ImageDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(ImageDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn