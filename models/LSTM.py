import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Implementation of LSTM
Reference: SIGN LANGUAGE RECOGNITION WITH LONG SHORT-TERM MEMORY
"""
class LSTM(nn.Module):
    def __init__(self, lstm_input_size=512, lstm_hidden_size=512, lstm_num_layers=3,
                num_classes=100, hidden1=256, drop_p=0.0):
        super(LSTM, self).__init__()
        # network params
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.num_classes = num_classes
        self.hidden1 = hidden1
        self.drop_p = drop_p

        # network architecture
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
        )
        self.drop = nn.Dropout2d(p=self.drop_p)
        self.fc1 = nn.Linear(self.lstm_hidden_size, self.hidden1)
        self.fc2 = nn.Linear(self.hidden1, self.num_classes)

    def forward(self, x):
        # LSTM
        # use faster code paths
        self.lstm.flatten_parameters()
        # print(x.shape)
        # batch first: (batch, seq, feature)
        out, (h_n, c_n) = self.lstm(x, None)
        # MLP
        # out: (batch, seq, feature), choose the last time step
        out = F.relu(self.fc1(out[:, -1, :]))
        out = F.dropout(out, p=self.drop_p, training=self.training)
        out = self.fc2(out)

        return out


# Test
if __name__ == '__main__':
    import sys
    sys.path.append("..")
    from dataset import CSL_Skeleton
    selected_joints = ['HANDLEFT', 'HANDRIGHT', 'ELBOWLEFT', 'ELBOWRIGHT']
    lstm_input_size = len(selected_joints)*2
    dataset = CSL_Skeleton(data_path="/home/haodong/Data/CSL_Isolated_1/xf500_body_depth_txt",
        label_path="/home/haodong/Data/CSL_Isolated_1/dictionary.txt", selected_joints=selected_joints)
    lstm = LSTM(lstm_input_size=lstm_input_size)
    print(lstm(dataset[0]['images'].unsqueeze(0)))
