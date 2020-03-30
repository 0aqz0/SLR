import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

import os,inspect,sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir)
from Attention import LSTMAttentionBlock

"""
Implementation of CNN+LSTM.
"""
class CRNN(nn.Module):
    def __init__(self, sample_size=256, sample_duration=16, num_classes=100,
                lstm_hidden_size=512, lstm_num_layers=1):
        super(CRNN, self).__init__()
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.num_classes = num_classes

        # network params
        self.ch1, self.ch2, self.ch3, self.ch4 = 64, 128, 256, 512
        self.k1, self.k2, self.k3, self.k4 = (7, 7), (3, 3), (3, 3), (3, 3)
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (1, 1), (1, 1), (1, 1)
        self.p1, self.p2, self.p3, self.p4 = (0, 0), (0, 0), (0, 0), (0, 0)
        self.d1, self.d2, self.d3, self.d4 = (1, 1), (1, 1), (1, 1), (1, 1)
        self.lstm_input_size = self.ch4
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        # network architecture
        # in_channels=3 for rgb
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.p1, dilation=self.d1),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch1, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.p2, dilation=self.d2),
            nn.BatchNorm2d(self.ch2, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch2, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.p3, dilation=self.d3),
            nn.BatchNorm2d(self.ch3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.ch3, out_channels=self.ch3, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4, padding=self.p4, dilation=self.d4),
            nn.BatchNorm2d(self.ch4, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.ch4, out_channels=self.ch4, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(self.lstm_hidden_size, self.num_classes)

    def forward(self, x):
        # CNN
        cnn_embed_seq = []
        # print(x.shape)
        # x: (batch_size, channel, t, h, w)
        for t in range(x.size(2)):
            # Conv
            out = self.conv1(x[:, :, t, :, :])
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.conv4(out)
            # print(out.shape)
            out = out.view(out.size(0), -1)
            cnn_embed_seq.append(out)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        # print(cnn_embed_seq.shape)
        # batch first
        cnn_embed_seq = cnn_embed_seq.transpose_(0, 1)

        # LSTM
        # use faster code paths
        self.lstm.flatten_parameters()
        out, (h_n, c_n) = self.lstm(cnn_embed_seq, None)
        # MLP
        # out: (batch, seq, feature), choose the last time step
        out = self.fc1(out[:, -1, :])

        return out


"""
Implementation of Resnet+LSTM
"""
class ResCRNN(nn.Module):
    def __init__(self, sample_size=256, sample_duration=16, num_classes=100,
                lstm_hidden_size=512, lstm_num_layers=1, arch="resnet18",
                attention=False):
        super(ResCRNN, self).__init__()
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.num_classes = num_classes

        # network params
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.attention = attention

        # network architecture
        if arch == "resnet18":
            resnet = models.resnet18(pretrained=True)
        elif arch == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif arch == "resnet50":
            resnet = models.resnet50(pretrained=True)
        elif arch == "resnet101":
            resnet = models.resnet101(pretrained=True)
        elif arch == "resnet152":
            resnet = models.resnet152(pretrained=True)
        # delete the last fc layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.lstm = nn.LSTM(
            input_size=resnet.fc.in_features,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
        )
        if self.attention:
            self.attn_block = LSTMAttentionBlock(hidden_size=self.lstm_hidden_size)
        self.fc1 = nn.Linear(self.lstm_hidden_size, self.num_classes)

    def forward(self, x):
        # CNN
        cnn_embed_seq = []
        # x: (batch_size, channel, t, h, w)
        for t in range(x.size(2)):
            # with torch.no_grad():
            out = self.resnet(x[:, :, t, :, :])
            # print(out.shape)
            out = out.view(out.size(0), -1)
            cnn_embed_seq.append(out)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        # print(cnn_embed_seq.shape)
        # batch first
        cnn_embed_seq = cnn_embed_seq.transpose_(0, 1)

        # LSTM
        # use faster code paths
        self.lstm.flatten_parameters()
        out, (h_n, c_n) = self.lstm(cnn_embed_seq, None)
        # MLP
        if self.attention:
            out = self.fc1(self.attn_block(out))
        else:
            # out: (batch, seq, feature), choose the last time step
            out = self.fc1(out[:, -1, :])

        return out


# Test
if __name__ == '__main__':
    import sys
    sys.path.append("..")
    import torchvision.transforms as transforms
    from dataset import CSL_Isolated
    sample_size = 128
    sample_duration = 16
    num_classes = 500
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]), transforms.ToTensor()])
    dataset = CSL_Isolated(data_path="/home/haodong/Data/CSL_Isolated/color_video_125000",
        label_path="/home/haodong/Data/CSL_Isolated/dictionary.txt", frames=sample_duration,
        num_classes=num_classes, transform=transform)
    # crnn = CRNN()
    crnn = ResCRNN(sample_size=sample_size, sample_duration=sample_duration, num_classes=num_classes, arch="resnet152")
    print(crnn(dataset[0]['data'].unsqueeze(0)))
