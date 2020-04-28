import os
import sys
from datetime import datetime
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from models.ConvLSTM import CRNN, ResCRNN
from dataset import CSL_Isolated
from train import train_epoch
from validation import val_epoch

# Path setting
data_path = "/home/haodong/Data/CSL_Isolated/color_video_125000"
label_path = "/home/haodong/Data/CSL_Isolated/dictionary.txt"
model_path = "/home/haodong/Data/cnnlstm_models"
log_path = "log/cnnlstm_{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now())
sum_path = "runs/slr_cnnlstm_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

# Log to file & tensorboard writer
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
logger = logging.getLogger('SLR')
logger.info('Logging to file...')
writer = SummaryWriter(sum_path)

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]="2"
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
epochs = 200
batch_size = 16
learning_rate = 1e-4
weight_decay = 1e-5
log_interval = 20
sample_size = 128
sample_duration = 16
num_classes = 100
lstm_hidden_size = 512
lstm_num_layers = 1
attention = False

# Train with Conv+LSTM
if __name__ == '__main__':
    # Load data
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    train_set = CSL_Isolated(data_path=data_path, label_path=label_path, frames=sample_duration,
        num_classes=num_classes, train=True, transform=transform)
    val_set = CSL_Isolated(data_path=data_path, label_path=label_path, frames=sample_duration,
        num_classes=num_classes, train=False, transform=transform)
    logger.info("Dataset samples: {}".format(len(train_set)+len(val_set)))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    # Create model
    # model = CRNN(sample_size=sample_size, sample_duration=sample_duration, num_classes=num_classes,
    #             lstm_hidden_size=lstm_hidden_size, lstm_num_layers=lstm_num_layers).to(device)
    model = ResCRNN(sample_size=sample_size, sample_duration=sample_duration, num_classes=num_classes,
                lstm_hidden_size=lstm_hidden_size, lstm_num_layers=lstm_num_layers, attention=attention).to(device)
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
        train_epoch(model, criterion, optimizer, train_loader, device, epoch, logger, log_interval, writer)

        # Validate the model
        val_epoch(model, criterion, val_loader, device, epoch, logger, writer)

        # Save model
        torch.save(model.state_dict(), os.path.join(model_path, "slr_convlstm_epoch{:03d}.pth".format(epoch+1)))
        logger.info("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

    logger.info("Training Finished".center(60, '#'))
