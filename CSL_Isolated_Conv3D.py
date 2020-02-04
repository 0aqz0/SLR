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
from models.Conv3D import CNN3D
from dataset import CSL_Isolated
from train import train
from test import test

# Path setting
data_path = "/home/aistudio/data/data20273/CSL_Isolated_125000"
label_path = "/home/aistudio/data/data20273/CSL_Isolated_125000/dictionary.txt"
model_path = "."
log_path = "log/{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now())
sum_path = "runs/slr_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

# Log to file & tensorboard writer
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
logger = logging.getLogger('SLR')
logger.info('Logging to file...')
# writer = SummaryWriter(sum_path)

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
num_classes = 100
epochs = 30
batch_size = 32
learning_rate = 1e-4
log_interval = 20
img_d, img_h, img_w = 16, 128, 128
drop_p = 0.0
hidden1, hidden2 = 512, 256

# Train with 3DCNN
if __name__ == '__main__':
    # Load data
    transform = transforms.Compose([transforms.Resize([img_h, img_w]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    dataset = CSL_Isolated(data_path=data_path, label_path=label_path, frames=img_d, num_classes=num_classes, transform=transform)
    trainset, testset = random_split(dataset, [int(0.8*len(dataset)), int(0.2*len(dataset))])
    logger.info("Dataset samples: {}".format(len(dataset)))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    # Create model
    model = CNN3D(img_depth=img_d, img_height=img_h, img_width=img_w, drop_p=drop_p,
                hidden1=hidden1, hidden2=hidden2, num_classes=num_classes).to(device)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Start training
    logger.info("Training Started".center(60, '#'))
    for epoch in range(epochs):
        # Train the model
        train(model, criterion, optimizer, trainloader, device, epoch, logger, log_interval)

        # Test the model
        test(model, criterion, testloader, device, epoch, logger)

        # Save model
        torch.save(model.state_dict(), os.path.join(model_path, "slr_cnn3d_epoch{}.pth".format(epoch+1)))
        logger.info("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

    logger.info("Training Finished".center(60, '#'))
