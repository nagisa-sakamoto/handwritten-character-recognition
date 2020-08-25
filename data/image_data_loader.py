import torch
from torch.utils.data import DataLoader

def _collate_img_fn(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    minibatch_size = len(batch)
    inputs = torch.zeros(minibatch_size, longest_sample.size(0), longest_sample.size(1))
    input_sizes = torch.IntTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        feature = sample[0]
        target = sample[1]
        seq_length = feature.size(1)
        inputs[x].narrow(1, 0, seq_length).copy_(feature)
        input_sizes[x] = seq_length
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_sizes, target_sizes

def _collate_strided_img_fn(batch):
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
        self.collate_fn = _collate_img_fn
