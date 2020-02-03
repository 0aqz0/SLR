import os
import sys
from datetime import datetime
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
log_path = "./log{:_%Y-%m-%d_%H-%M-%S}.txt".format(datetime.now())
sum_path = "runs/slr{:_%Y-%m-%d_%H-%M-%S}".format(datetime.now())

# Log to file & tensorboard writer
log_to_file = True
if log_to_file:
    log = open(log_path, "w")
    sys.stdout = log
    print("Logging to file...")
# logger = SummaryWriter(sum_path)

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
num_classes = 500
epochs = 60
batch_size = 32
learning_rate = 1e-4
log_interval = 10
img_d, img_h, img_w = 25, 128, 96
drop_p = 0.0
hidden1, hidden2 = 512, 256


if __name__ == '__main__':
    # Train with 3DCNN
    # Load data
    transform = transforms.Compose([transforms.Resize([img_h, img_w]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    dataset = CSL_Isolated(data_path=data_path, label_path=label_path, frames=img_d, transform=transform)
    trainset, testset = random_split(dataset, [int(0.8*len(dataset)), int(0.2*len(dataset))])
    print("Dataset samples: {}".format(len(dataset)))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # Create model
    model = CNN3D(img_depth=img_d, img_height=img_h, img_width=img_w, drop_p=drop_p,
                hidden1=hidden1, hidden2=hidden2, num_classes=num_classes).to(device)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Start training
    print("Training Started".center(60, '#'))
    for epoch in range(epochs):
        # Train the model
        train(model, criterion, optimizer, trainloader, device, epoch, log_interval)

        # Test the model
        test(model, criterion, testloader, device, epoch)

        # Save model
        torch.save(model.state_dict(), os.path.join(model_path, "slr_cnn3d_epoch{}.pth".format(epoch+1)))
        print("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

    print("Training Finished".center(60, '#'))
