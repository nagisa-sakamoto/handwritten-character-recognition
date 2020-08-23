import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CnnLstmE2EModel(nn.Module):

    def __init__(self, targets,\
                input_size=96,\
                rnn_dim=256, num_rnn_layers=2):
        super(CnnLstmE2EModel, self).__init__()
        self.targets = targets
        self.target_size = len(targets)
        self.rnn_dim = rnn_dim
        self.num_rnn_layers = num_rnn_layers
        self.conv = nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=0),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2)
        )
        cnn_output_size = self.get_conv_output_lengths(torch.IntTensor([input_size]), axis=0)
        rnn_input_size = cnn_output_size[0] * 64
        self.rnn = nn.LSTM(input_size=rnn_input_size,\
                        hidden_size=self.rnn_dim,\
                        num_layers=self.num_rnn_layers,\
                        batch_first=True,\
                        bidirectional=True)
        self.fc = nn.Linear(self.rnn_dim * 2, self.target_size)

    def get_target(self, index=0):
        return self.targets[index]

    def get_target_seq(self, index=[]):
        target_seq = ''
        for i in index:
            target_seq += self.targets[i]
        return target_seq
    
    def forward(self, x, input_lengths):
        output_lengths = self.get_conv_output_lengths(input_lengths)
        x.unsqueeze_(1)
        x = self.conv(x)
        batch_size, C, H, W = x.size()
        x = x.view(batch_size, C*H, W) # batchN x featureDim x frameN
        x = x.transpose(1, 2).contiguous() # batchN x frameN x featureDim
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths, batch_first=True)
        x, _ = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        y = self.fc(x)
        return y, output_lengths

    def get_conv_output_lengths(self, input_lengths, axis=1):
        seq_len = input_lengths
        for m in self.conv.modules():
            if type(m) == nn.Conv2d:
                seq_len = torch.floor_divide((seq_len + 2 * m.padding[axis] - m.dilation[axis] * (m.kernel_size[axis] - 1) - 1), m.stride[axis]) + 1
            elif type(m) == nn.MaxPool2d:
                seq_len = torch.floor_divide((seq_len + 2 * m.padding - m.dilation * (m.kernel_size - 1) - 1), m.stride) + 1
        return seq_len

