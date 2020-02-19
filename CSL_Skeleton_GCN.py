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
from models.GCN import GCN
from dataset import CSL_Skeleton
from train import train
from test import test

# Path setting
data_path = "/home/haodong/Data/CSL_Isolated_1/xf500_body_depth_txt"
label_path = "/home/haodong/Data/CSL_Isolated_1/dictionary.txt"
model_path = "/home/haodong/Data/gcn_models"
log_path = "log/gcn_{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now())
sum_path = "runs/slr_gcn_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

# Log to file & tensorboard writer
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
logger = logging.getLogger('SLR')
logger.info('Logging to file...')
writer = SummaryWriter(sum_path)

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
epochs = 200
batch_size = 32
learning_rate = 1e-5
log_interval = 100
num_classes = 500
in_channels = 2
sample_duration = 16
selected_joints = None
split_to_channels = True

# Train with GCN
if __name__ == '__main__':
    # Load data
    transform = None # TODO
    trainset = CSL_Skeleton(data_path=data_path, label_path=label_path, frames=sample_duration, num_classes=num_classes,
        selected_joints=selected_joints, split_to_channels=split_to_channels, train=True, transform=transform)
    testset = CSL_Skeleton(data_path=data_path, label_path=label_path, frames=sample_duration, num_classes=num_classes,
        selected_joints=selected_joints, split_to_channels=split_to_channels, train=False, transform=transform)
    logger.info("Dataset samples: {}".format(len(trainset)+len(testset)))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # Create model
    model = GCN(in_channels=in_channels, num_class=num_classes, graph_args={'layout': 'ntu-rgb+d'},
                 edge_importance_weighting=True).to(device)
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
        train(model, criterion, optimizer, trainloader, device, epoch, logger, log_interval, writer)

        # Test the model
        test(model, criterion, testloader, device, epoch, logger, writer)

        # Save model
        torch.save(model.state_dict(), os.path.join(model_path, "slr_gcn_epoch{:03d}.pth".format(epoch+1)))
        logger.info("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

    logger.info("Training Finished".center(60, '#'))
