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


# Test
if __name__ == '__main__':
    import torchvision.transforms as transforms
    from dataset import CSL_Isolated
    transform = transforms.Compose([transforms.Resize([128, 96]), transforms.ToTensor()])
    dataset = CSL_Isolated(data_path="/home/aistudio/data/data20273/CSL_Isolated_125000",
        label_path='/home/aistudio/data/data20273/CSL_Isolated_125000/dictionary.txt', transform=transform)
    cnn3d = CNN3D()
    # print(dataset[0]['images'].shape)
    print(cnn3d(dataset[0]['images'].unsqueeze(0)))
