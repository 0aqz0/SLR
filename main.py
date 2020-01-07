import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from models import CNN3D
from dataset import CSL_Dataset

PATH = './slr_cnn3d.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparams
batch_size = 16
learning_rate = 1e-3
max_epochs = 16

if __name__ == '__main__':
    # Train with 3DCNN
    transform = transforms.Compose([transforms.Resize([64, 48]), transforms.ToTensor()])
    dataset = CSL_Dataset(data_path="/home/ddq/Data/origin/S500_color_video",
        label_path='/home/ddq/Data/origin/dictionary.txt', transform=transform)
    print("Dataset samples: {}".format(len(dataset)))
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    cnn3d = CNN3D()
    cnn3d.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn3d.parameters(), lr=learning_rate)
    writer = SummaryWriter('runs/slr')
    print("Training Started".center(40, '#'))
    for epoch in range(max_epochs):
        running_loss = 0.0
        running_acc = 0.0
        iteration_count = 0

        for i, data in enumerate(trainloader):
            # get the inputs and labels
            inputs, labels = data['images'].to(device), data['label'].to(device)

            # zero the gradient
            optimizer.zero_grad()

            # forward & backward & optimize
            outputs = cnn3d(inputs)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()

            # calculate the loss
            running_loss += loss.item()

            # calculate the acc
            acc = torch.sum((torch.argmax(outputs, 1) == labels)) / batch_size
            # print(outputs, labels)
            # print((torch.argmax(outputs, 1)))
            # count iteration
            # iteration_count += 1

            if i % 4 == 3:
                print("epoch {} | iteration {:5d} | Loss {} | Acc {}".format(epoch+1, i+1, running_loss/4, acc))
                writer.add_scalar('training loss', running_loss/4, i+1)
                running_loss = 0.0

        # print log file
        # print("Epoch {} | Loss {} | Acc {}".format(epoch, running_loss / iteration, running_acc / iteration))

    print("Training Finished".center(40, '#'))

    # Save model
    torch.save(cnn3d.state_dict(), PATH)
    print("Model Saved".center(40, '#'))