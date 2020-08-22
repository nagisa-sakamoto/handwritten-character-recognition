import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class CnnLstmModel(nn.Module):

    def __init__(self, targets, input_map_size=(96, 96), rnn_dim=256, num_rnn_layers=2):
        super(CnnLstmModel, self).__init__()
        self.input_map_size = input_map_size
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
        rnn_input_size = 3 * 3 * 64
        self.rnn = nn.LSTM(input_size=rnn_input_size,\
                        hidden_size=self.rnn_dim,\
                        num_layers=self.num_rnn_layers,\
                        batch_first=True,\
                        bidirectional=True)
        self.fc = nn.Linear(self.rnn_dim * 2, self.target_size)
    

    def get_input_map_size(self):
        return self.input_map_size

    def get_target(self, index=0):
        return self.targets[index]

    def get_target_seq(self, index=[]):
        target_seq = ''
        for i in index:
            target_seq += self.targets[i]
        return target_seq
    
    def forward(self, x, input_lengths):
        x.unsqueeze_(2)
        batch_size, seq_length, C, H, W = x.size()
        x = x.view(batch_size * seq_length, C, H, W) # batchN*frameN x C x H x W
        x = self.conv(x)
        x = x.view(batch_size, seq_length, -1) # batchN x frameN x featureDim
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)
        x, _ = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        y = self.fc(x)
        return y


