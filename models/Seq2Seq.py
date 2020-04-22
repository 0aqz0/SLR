import torch
import torch.nn as nn
import torchvision.models as models
import random

import os,inspect,sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir)
from ConvLSTM import ResCRNN

"""
Implementation of Sequence to Sequence Model
Encoder: encode video spatial and temporal dynamics e.g. CNN+LSTM
Decoder: decode the compressed info from encoder
"""
class Encoder(nn.Module):
    def __init__(self, lstm_hidden_size=512, arch="resnet18"):
        super(Encoder, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size

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
            batch_first=True,
        )

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
        # batch first
        cnn_embed_seq = cnn_embed_seq.transpose_(0, 1)

        # LSTM
        # use faster code paths
        self.lstm.flatten_parameters()
        out, (h_n, c_n) = self.lstm(cnn_embed_seq, None)

        # num_layers * num_directions = 1
        return out, (h_n.squeeze(0), c_n.squeeze(0))


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim+enc_hid_dim, dec_hid_dim)
        self.fc = nn.Linear(emb_dim+enc_hid_dim+dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, context):
        # input(batch_size): last prediction
        # hidden(batch_size, dec_hid_dim): decoder last hidden state
        # cell(batch_size, dec_hid_dim): decoder last cell state
        # context(batch_size, enc_hid_dim): context vector
        # print(input.shape, hidden.shape, cell.shape, context.shape)
        # expand dim to (1, batch_size)
        input = input.unsqueeze(0)

        # embedded(1, batch_size, emb_dim): embed last prediction word
        embedded = self.dropout(self.embedding(input))

        # rnn_input(1, batch_size, emb_dim+enc_hide_dim): concat embedded and context 
        rnn_input = torch.cat((embedded, context.unsqueeze(0)), dim=2)

        # output(seq_len, batch, num_directions * hidden_size)
        # hidden(num_layers * num_directions, batch, hidden_size)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))

        # hidden(batch_size, dec_hid_dim)
        # cell(batch_size, dec_hid_dim)
        # embedded(1, batch_size, emb_dim)
        hidden = hidden.squeeze(0)
        cell = cell.squeeze(0)
        embedded = embedded.squeeze(0)

        # prediction
        prediction = self.fc(torch.cat((embedded, context, hidden), dim=1))

        return prediction, (hidden, cell)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, imgs, target, teacher_forcing_ratio=0.5):
        # imgs: (batch_size, channels, T, H, W)
        # target: (batch_size, trg len)
        batch_size = imgs.shape[0]
        trg_len = target.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs(batch, seq_len, hidden_size): all hidden states of input sequence
        encoder_outputs, (hidden, cell) = self.encoder(imgs)

        # compute context vector
        context = encoder_outputs.mean(dim=1)

        # first input to the decoder is the <sos> tokens
        input = target[:,0]

        for t in range(1, trg_len):
            # decode
            output, (hidden, cell) = self.decoder(input, hidden, cell, context)

            # store prediction
            outputs[t] = output

            # decide whether to do teacher foring
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token
            top1 = output.argmax(1)

            # apply teacher forcing
            input = target[:,t] if teacher_force else top1

        return outputs


# Test
if __name__ == '__main__':
    # test encoder
    encoder = Encoder(lstm_hidden_size=512)
    # imgs = torch.randn(16, 3, 8, 128, 128)
    # print(encoder(imgs))

    # test decoder
    decoder = Decoder(output_dim=500, emb_dim=256, enc_hid_dim=512, dec_hid_dim=512, dropout=0.5)
    # input = torch.LongTensor(16).random_(0, 500)
    # hidden = torch.randn(16, 512)
    # cell = torch.randn(16, 512)
    # context = torch.randn(16, 512)
    # print(decoder(input, hidden, cell, context))

    # test seq2seq
    device = torch.device("cpu")
    seq2seq = Seq2Seq(encoder=encoder, decoder=decoder, device=device)
    imgs = torch.randn(16, 3, 8, 128, 128)
    target = torch.LongTensor(16, 8).random_(0, 500)
    print(seq2seq(imgs, target).argmax(dim=2).permute(1,0)) # batch first
