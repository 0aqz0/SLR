import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from models import CNN3D
from dataset import CSL_Dataset

# Path setting
data_path = "/home/ddq/Data/CSL_Dataset/S500_color_video"
label_path = "/home/ddq/Data/CSL_Dataset/dictionary.txt"
model_path = "./slr_cnn3d.pth"
log_path = "./log.txt"

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
num_classes = 100
epochs = 15
batch_size = 30
learning_rate = 1e-4
log_interval = 10
img_d, img_h, img_w = 25, 128, 96
drop_p = 0.0
hidden1, hidden2 = 256, 256


if __name__ == '__main__':
    # Train with 3DCNN
    # Load data
    transform = transforms.Compose([transforms.Resize([img_h, img_w]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    dataset = CSL_Dataset(data_path=data_path, label_path=label_path, frames=img_d, transform=transform)
    print("Dataset samples: {}".format(len(dataset)))
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # Create model
    cnn3d = CNN3D(img_depth=img_d, img_height=img_h, img_width=img_w, drop_p=drop_p,
                hidden1=hidden1, hidden2=hidden2, num_classes=num_classes).to(device)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        cnn3d = nn.DataParallel(cnn3d)
    # Create loss criterion & optimizer & log writer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn3d.parameters(), lr=learning_rate)
    # writer = SummaryWriter('runs/slr')
    # log = open(log_path, "w")

    # Start training
    print("Training Started".center(80, '#'))
    for epoch in range(epochs):
        # Set trainning mode
        cnn3d.train()

        running_loss = 0.0

        for i, data in enumerate(trainloader):
            # get the inputs and labels
            inputs, labels = data['images'].to(device), data['label'].to(device)

            optimizer.zero_grad()
            # forward & backward & optimize
            outputs = cnn3d(inputs)
            loss = F.cross_entropy(outputs, labels.squeeze())

            # compute the loss
            running_loss += loss.item()

            # compute the accuracy
            prediction = torch.max(outputs, 1)[1]
            score = accuracy_score(labels.squeeze().cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())

            loss.backward()
            optimizer.step()

            if (i + 1) % log_interval == 0:
                print("epoch {:3d} | iteration {:5d} | Loss {:.6f} | Acc {:.2f}%".format(epoch+1, i+1, loss.item(), score*100))
                # writer.add_scalar('training loss', running_loss/4, i+1)
                running_loss = 0.0
                acc = 0.0

    print("Training Finished".center(80, '#'))

    # Save model
    torch.save(cnn3d.state_dict(), model_path)
    print("Model Saved".center(80, '#'))