import torch
import torch.nn as nn

"""
Implementation of Sequence to Sequence Model
Encoder: encode video spatial and temporal dynamics e.g. CNN+LSTM
Decoder: decode the compressed info from encoder
"""
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
        # context(batch_size, enc_hid_dim): encoder final hidden state
        
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

        return prediction, hidden, cell


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
        trg_len = target.shape[0]


# Test
if __name__ == '__main__':
    # test decoder
    decoder = Decoder(output_dim=500, emb_dim=256, enc_hid_dim=512, dec_hid_dim=512, dropout=0.5)
    input = torch.LongTensor(16).random_(0, 500)
    hidden = torch.randn(16, 512)
    cell = torch.randn(16, 512)
    context = torch.randn(16, 512)
    print(decoder(input, hidden, cell, context))
