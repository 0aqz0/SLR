import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Attention blocks
Reference: Learn To Pay Attention
"""
class ProjectorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        return self.op(x)


class ProjectorBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProjectorBlock3D, self).__init__()
        self.op = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        return self.op(x)


class LinearAttentionBlock(nn.Module):
    def __init__(self, in_channels, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_channels, out_channels=1,
            kernel_size=1, padding=0, bias=False)

    def forward(self, l, g):
        N, C, H, W = l.size()
        c = self.op(l+g) # (batch_size,1,H,W)
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,H,W)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # (batch_size,C)
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,H,W), g


class LinearAttentionBlock3D(nn.Module):
    def __init__(self, in_channels, normalize_attn=True):
        super(LinearAttentionBlock3D, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv3d(in_channels=in_channels, out_channels=1,
            kernel_size=1, padding=0, bias=False)

    def forward(self, l, g):
        N, C, T, H, W = l.size()
        c = self.op(l+g) # (batch_size,1,T,H,W)
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,T,H,W)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # (batch_size,C)
        else:
            g = F.adaptive_avg_pool3d(g, (1,1,1)).view(N,C)
        return c.view(N,1,T,H,W), g

"""
Dense attention block
Reference: https://github.com/philipperemy/keras-attention-mechanism
"""
class LSTMAttentionBlock(nn.Module):
    def __init__(self, hidden_size):
        super(LSTMAttentionBlock, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)

    def forward(self, hidden_states):
        # (batch_size, time_steps, hidden_size)
        score_first_part = self.fc1(hidden_states)
        # (batch_size, hidden_size)
        h_t = hidden_states[:,-1,:]
        # (batch_size, time_steps)
        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(score, dim=1)
        # (batch_size, hidden_size)
        context_vector = torch.bmm(hidden_states.permute(0,2,1), attention_weights.unsqueeze(2)).squeeze(2)
        # (batch_size, hidden_size*2)
        pre_activation = torch.cat((context_vector, h_t), dim=1)
        # (batch_size, hidden_size)
        attention_vector = self.fc2(pre_activation)
        attention_vector = torch.tanh(attention_vector)

        return attention_vector

# Test
if __name__ == '__main__':
    # 2d block
    attention_block = LinearAttentionBlock(in_channels=3)
    l = torch.randn(16, 3, 128, 128)
    g = torch.randn(16, 3, 128, 128)
    print(attention_block(l, g))
    # 3d block
    attention_block_3d = LinearAttentionBlock3D(in_channels=3)
    l = torch.randn(16, 3, 16, 128, 128)
    g = torch.randn(16, 3, 16, 128, 128)
    print(attention_block_3d(l, g))
    # LSTM block
    attention_block_lstm = LSTMAttentionBlock(hidden_size=256)
    hidden_states = torch.randn(32, 16, 256)
    print(attention_block_lstm(hidden_states).shape)
