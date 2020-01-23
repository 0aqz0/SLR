import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
Implementation of 3D CNN.
"""
class CNN3D(nn.Module):
    def __init__(self, img_depth=25, img_height=128, img_width=96, drop_p=0.0, hidden1=512, hidden2=256, num_classes=100):
        super(CNN3D, self).__init__()
        self.img_depth = img_depth
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes

        # network params
        self.ch1, self.ch2, self.ch3 = 32, 48, 48
        self.k1, self.k2, self.k3 = (5,7,7), (3,7,7), (3,5,5)
        self.s1, self.s2, self.s3 = (2,2,2), (2,2,2), (2,2,2)
        self.p1, self.p2, self.p3 = (0,0,0), (0,0,0), (0,0,0)
        self.d1, self.d2, self.d3 = (1,1,1), (1,1,1), (1,1,1)
        self.hidden1, self.hidden2 = hidden1, hidden2
        self.drop_p = drop_p
        self.pool_k, self.pool_s, self.pool_p, self.pool_d = (1,2,2), (1,2,2), (0,0,0), (1,1,1)
        # Conv1
        self.conv1_output_shape = self.compute_output_shape(self.img_depth, self.img_height, 
            self.img_width, self.k1, self.s1, self.p1, self.d1)
        # self.conv1_output_shape = self.compute_output_shape(self.conv1_output_shape[0], self.conv1_output_shape[1], 
        #     self.conv1_output_shape[2], self.pool_k, self.pool_s, self.pool_p, self.pool_d)
        # Conv2
        self.conv2_output_shape = self.compute_output_shape(self.conv1_output_shape[0], self.conv1_output_shape[1], 
            self.conv1_output_shape[2], self.k2, self.s2, self.p2, self.d2)
        # self.conv2_output_shape = self.compute_output_shape(self.conv2_output_shape[0], self.conv2_output_shape[1], 
        #     self.conv2_output_shape[2], self.pool_k, self.pool_s, self.pool_p, self.pool_d)
        # Conv3
        self.conv3_output_shape = self.compute_output_shape(self.conv2_output_shape[0], self.conv2_output_shape[1],
            self.conv2_output_shape[2], self.k3, self.s3, self.p3, self.d3)
        # print(self.conv1_output_shape, self.conv2_output_shape, self.conv3_output_shape)

        # network architecture
        # in_channels=1 for grayscale
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=self.ch1, kernel_size=self.k1,
            stride=self.s1, padding=self.p1, dilation=self.d1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2,
            stride=self.s2, padding=self.p2, dilation=self.d2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.conv3 = nn.Conv3d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3,
            stride=self.s3, padding=self.p3, dilation=self.d3)
        self.bn3 = nn.BatchNorm3d(self.ch3)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(p=self.drop_p)
        self.pool = nn.MaxPool3d(kernel_size=self.pool_k)
        self.fc1 = nn.Linear(self.ch3 * self.conv3_output_shape[0] * self.conv3_output_shape[1] * self.conv3_output_shape[2], self.hidden1)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2)
        self.fc3 = nn.Linear(self.hidden2, self.num_classes)

    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.pool(x)
        x = self.drop(x)
        # Conv2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # x = self.pool(x)
        x = self.drop(x)
        # Conv3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.drop(x)
        # MLP
        # print(x.shape)
        # x.size(0) ------ batch_size
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)

        return x

    def compute_output_shape(self, D_in, H_in, W_in, k, s, p, d):
        # Conv
        D_out = np.floor((D_in + 2*p[0] - d[0]*(k[0] - 1) - 1)/s[0] + 1).astype(int)
        H_out = np.floor((H_in + 2*p[1] - d[1]*(k[1] - 1) - 1)/s[1] + 1).astype(int)
        W_out = np.floor((W_in + 2*p[2] - d[2]*(k[2] - 1) - 1)/s[2] + 1).astype(int)
        
        return D_out, H_out, W_out

"""
Implementation of CNN+LSTM.
"""
class CRNN(nn.Module):
    def __init__(self, img_depth=25, img_height=128, img_width=96, drop_p=0.0, hidden1=512, hidden2=256, hidden3=256,
                cnn_embed_dim=512, lstm_hidden_size=512, lstm_num_layers=3, num_classes=100):
        super(CRNN, self).__init__()
        self.img_depth = img_depth
        self.img_height = img_height
        self.img_width = img_width
        self.cnn_embed_dim = cnn_embed_dim
        self.lstm_input_size = self.cnn_embed_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.num_classes = num_classes

        # network params
        self.ch1, self.ch2, self.ch3, self.ch4 = 32, 64, 128, 256
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)
        self.p1, self.p2, self.p3, self.p4 = (0, 0), (0, 0), (0, 0), (0, 0)
        self.d1, self.d2, self.d3, self.d4 = (1, 1), (1, 1), (1, 1), (1, 1)
        self.hidden1, self.hidden2, self.hidden3 = hidden1, hidden2, hidden3
        self.drop_p = drop_p
        # compute output shape
        self.conv1_output_shape = self.compute_output_shape(self.img_height, self.img_width, self.k1, self.s1, self.p1, self.d1)
        self.conv2_output_shape = self.compute_output_shape(self.conv1_output_shape[0], self.conv1_output_shape[1], self.k2, self.s2, self.p2, self.d2)
        self.conv3_output_shape = self.compute_output_shape(self.conv2_output_shape[0], self.conv2_output_shape[1], self.k3, self.s3, self.p3, self.d3)
        self.conv4_output_shape = self.compute_output_shape(self.conv3_output_shape[0], self.conv3_output_shape[1], self.k4, self.s4, self.p4, self.d4)
        # print(self.conv1_output_shape, self.conv2_output_shape, self.conv3_output_shape, self.conv4_output_shape)

        # network architecture
        # in_channels=1 for grayscale
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.p1, dilation=self.d1),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.p2, dilation=self.d2),
            nn.BatchNorm2d(self.ch2, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.p3, dilation=self.d3),
            nn.BatchNorm2d(self.ch3, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4, padding=self.p4, dilation=self.d4),
            nn.BatchNorm2d(self.ch4, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
        )
        self.drop = nn.Dropout2d(p=self.drop_p)
        self.fc1 = nn.Linear(self.ch4 * self.conv4_output_shape[0] * self.conv4_output_shape[1], self.hidden1)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2)
        self.fc3 = nn.Linear(self.hidden2, self.cnn_embed_dim)
        self.fc4 = nn.Linear(self.lstm_hidden_size, self.hidden3)
        self.fc5 = nn.Linear(self.hidden3, self.num_classes)

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
            # MLP
            out = out.view(out.size(0), -1)
            # print(out.shape)
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            out = F.dropout(out, p=self.drop_p, training=self.training)
            out = self.fc3(out)
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
        out = F.relu(self.fc4(out[:, -1, :]))
        out = F.dropout(out, p=self.drop_p, training=self.training)
        out = self.fc5(out)

        return out

    def compute_output_shape(self, H_in, W_in, k, s, p, d):
        # Conv
        H_out = np.floor((H_in + 2*p[0] - d[0]*(k[0] - 1) - 1)/s[0] + 1).astype(int)
        W_out = np.floor((W_in + 2*p[1] - d[1]*(k[1] - 1) - 1)/s[1] + 1).astype(int)

        return H_out, W_out


# Test
if __name__ == '__main__':
    import torchvision.transforms as transforms
    from dataset import CSL_Dataset
    transform = transforms.Compose([transforms.Resize([128, 96]), transforms.ToTensor()])
    dataset = CSL_Dataset(data_path="/home/haodong/Data/CSL_Dataset/S500_color_video",
        label_path='/home/haodong/Data/CSL_Dataset/dictionary.txt', transform=transform)
    # cnn3d = CNN3D()
    # # print(dataset[0]['images'].shape)
    # print(cnn3d(dataset[0]['images'].unsqueeze(0)))
    crnn = CRNN()
    print(crnn(dataset[0]['images'].unsqueeze(0)))
