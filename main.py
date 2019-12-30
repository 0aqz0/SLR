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
batch_size = 64
learning_rate = 1e-2
max_epochs = 1

if __name__ == '__main__':
    # Train with 3DCNN
    transform = transform = transforms.Compose([transforms.Resize([60, 64]), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = CSL_Dataset(data_path="/media/zjunlict/TOSHIBA1/dataset/SLR_dataset/S500_color_video",
        label_path='/media/zjunlict/TOSHIBA1/dataset/SLR_dataset/dictionary.txt', transform=transform)
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

        for i, data in enumerate(trainloader):
            # get the inputs and labels
            inputs, labels = data['images'].to(device), data['label'].to(device)
            # print(inputs.shape, labels.shape)

            # zero the gradient
            optimizer.zero_grad()

            # forward & backward & optimize
            outputs = cnn3d(inputs)
            # print(inputs.size(), labels.size(), outputs.size())
            # print(torch.max(outputs, 1)[1].size(), labels.squeeze().size())
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()

            # calculate the loss
            running_loss += loss.item()
            if i % 5 == 4:
                print("epoch {}, iteration {:5d}, Loss {}".format(epoch+1, i+1, running_loss/5))
                writer.add_scalar('training loss', running_loss/5, i+1)
                running_loss = 0.0

    print("Training Finished".center(40, '#'))

    # Save model
    torch.save(cnn3d.state_dict(), PATH)
    print("Model Saved".center(40, '#'))