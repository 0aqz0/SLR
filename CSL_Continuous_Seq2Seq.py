import os
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from dataset import CSL_Continuous
from models.Seq2Seq import Encoder, Decoder, Seq2Seq

# Path setting
data_path = "/home/haodong/Data/CSL_Continuous/color"
dict_path = "/home/haodong/Data/CSL_Continuous/dictionary.txt"
corpus_path = "/home/haodong/Data/CSL_Continuous/corpus.txt"
model_path = "/home/haodong/Data/seq2seq_models"
log_path = "log/seq2seq_{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now())
sum_path = "runs/slr_seq2seq_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

# Log to file & tensorboard writer
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
logger = logging.getLogger('SLR')
logger.info('Logging to file...')
writer = SummaryWriter(sum_path)

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]="3"
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
epochs = 200
batch_size = 8
learning_rate = 1e-4
weight_decay = 1e-5
sample_size = 128
sample_duration = 48
enc_hid_dim = 512
emb_dim = 256
dec_hid_dim = 512
dropout = 0.5
clip = 1
log_interval = 20

if __name__ == '__main__':
    # Load data
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    train_set = CSL_Continuous(data_path=data_path, dict_path=dict_path,
        corpus_path=corpus_path, frames=sample_duration, train=True, transform=transform)
    test_set = CSL_Continuous(data_path=data_path, dict_path=dict_path,
        corpus_path=corpus_path, frames=sample_duration, train=False, transform=transform)
    logger.info("Dataset samples: {}".format(len(train_set)+len(test_set)))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    # Create Model
    encoder = Encoder(lstm_hidden_size=enc_hid_dim, arch="resnet18").to(device)
    decoder = Decoder(output_dim=train_set.output_dim, emb_dim=emb_dim, enc_hid_dim=enc_hid_dim, dec_hid_dim=dec_hid_dim, dropout=dropout).to(device)
    model = Seq2Seq(encoder=encoder, decoder=decoder, device=device).to(device)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Start training
    logger.info("Training Started".center(60, '#'))
    for epoch in range(epochs):
        # Train the model
        model.train()
        losses = []
        all_trg = []
        all_pred = []

        for batch_idx, (imgs, target) in enumerate(train_loader):
            imgs = imgs.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            # forward
            outputs = model(imgs, target)

            # target: (batch_size, trg len)
            # outputs: (trg_len, batch_size, output_dim)
            # skip sos
            output_dim = outputs.shape[-1]
            outputs = outputs[1:].view(-1, output_dim)
            target = target.permute(1,0)[1:].reshape(-1)

            # compute the loss
            loss = criterion(outputs, target)
            losses.append(loss.item())

            # compute the accuracy
            prediction = torch.max(outputs, 1)[1]
            score = accuracy_score(target.cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())
            all_trg.extend(target)
            all_pred.extend(prediction)

            # backward & optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if (batch_idx + 1) % log_interval == 0:
                logger.info("epoch {:3d} | iteration {:5d} | Loss {:.6f} | Acc {:.2f}%".format(epoch+1, batch_idx+1, loss.item(), score*100))
        
        # Compute the average loss & accuracy
        training_loss = sum(losses)/len(losses)
        all_trg = torch.stack(all_trg, dim=0)
        all_pred = torch.stack(all_pred, dim=0)
        training_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
        # Log
        writer.add_scalars('Loss', {'train': training_loss}, epoch+1)
        writer.add_scalars('Accuracy', {'train': training_acc}, epoch+1)
        logger.info("Average Training Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch+1, training_loss, training_acc*100))

        # Test the model
        

        # Save model
        torch.save(model.state_dict(), os.path.join(model_path, "slr_seq2seq_epoch{:03d}.pth".format(epoch+1)))
        logger.info("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

    logger.info("Training Finished".center(60, '#'))
