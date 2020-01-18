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
from sklearn.metrics import accuracy_score
from models import CNN3D
from dataset import CSL_Dataset

# Path setting
data_path = "/home/ddq/Data/CSL_Dataset/S500_color_video"
label_path = "/home/ddq/Data/CSL_Dataset/dictionary.txt"
model_path = "."
log_path = "./log{:_%Y-%m-%d_%H-%M-%S}.txt".format(datetime.now())
sum_path = "runs/slr{:_%Y-%m-%d_%H-%M-%S}".format(datetime.now())

# Log to file & tensorboard writer
log_to_file = True
if log_to_file:
    log = open(log_path, "w")
    sys.stdout = log
    print("Logging to file...")
writer = SummaryWriter(sum_path)

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
    dataset = CSL_Dataset(data_path=data_path, label_path=label_path, frames=img_d, transform=transform)
    trainset, testset = random_split(dataset, [int(0.8*len(dataset)), int(0.2*len(dataset))])
    print("Dataset samples: {}".format(len(dataset)))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # Create model
    cnn3d = CNN3D(img_depth=img_d, img_height=img_h, img_width=img_w, drop_p=drop_p,
                hidden1=hidden1, hidden2=hidden2, num_classes=num_classes).to(device)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        cnn3d = nn.DataParallel(cnn3d)
    # Create loss criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn3d.parameters(), lr=learning_rate)

    # Start training
    print("Training Started".center(60, '#'))
    for epoch in range(epochs):
        # Set trainning mode
        cnn3d.train()
        losses = []
        all_label = []
        all_pred = []

        for batch_idx, data in enumerate(trainloader):
            # get the inputs and labels
            inputs, labels = data['images'].to(device), data['label'].to(device)

            optimizer.zero_grad()
            # forward
            outputs = cnn3d(inputs)

            # compute the loss
            loss = criterion(outputs, labels.squeeze())
            losses.append(loss.item())

            # compute the accuracy
            prediction = torch.max(outputs, 1)[1]
            all_label.extend(labels.squeeze())
            all_pred.extend(prediction)
            score = accuracy_score(labels.squeeze().cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())

            # backward & optimize
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % log_interval == 0:
                print("epoch {:3d} | iteration {:5d} | Loss {:.6f} | Acc {:.2f}%".format(epoch+1, batch_idx+1, loss.item(), score*100))

        # Compute the average loss & accuracy
        training_loss = sum(losses)/len(losses)
        all_label = torch.stack(all_label, dim=0)
        all_pred = torch.stack(all_pred, dim=0)
        training_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
        # Log
        writer.add_scalar('Loss/train', training_loss, epoch+1)
        writer.add_scalar('Accuracy/train', training_acc, epoch+1)
        print("Average Training Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch+1, training_loss, training_acc*100))

        # Test the model
        # Set testing mode
        cnn3d.eval()
        losses = []
        all_label = []
        all_pred = []

        with torch.no_grad():
            for batch_idx, data in enumerate(testloader):
                # get the inputs and labels
                inputs, labels = data['images'].to(device), data['label'].to(device)
                # forward
                outputs = cnn3d(inputs)
                # compute the loss
                loss = criterion(outputs, labels.squeeze())
                losses.append(loss.item())
                # collect labels & prediction
                prediction = torch.max(outputs, 1)[1]
                all_label.extend(labels.squeeze())
                all_pred.extend(prediction)
        # Compute the average loss & accuracy
        testing_loss = sum(losses)/len(losses)
        all_label = torch.stack(all_label, dim=0)
        all_pred = torch.stack(all_pred, dim=0)
        testing_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
        # Log
        writer.add_scalar('Loss/test', testing_loss, epoch+1)
        writer.add_scalar('Accuracy/test', testing_acc, epoch+1)
        print("Average Testing Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch+1, testing_loss, testing_acc*100))

        # Save model
        torch.save(cnn3d.state_dict(), os.path.join(model_path, "slr_cnn3d_epoch{}.pth".format(epoch+1)))
        print("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

    print("Training Finished".center(60, '#'))
