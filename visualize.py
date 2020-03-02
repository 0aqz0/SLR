import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, confusion_matrix
from dataset import CSL_Isolated
from models.Conv3D import resnet18, resnet34, resnet50
import numpy as np
import matplotlib.pyplot as plt

def get_label_and_pred(model, dataloader, device):
    all_label = []
    all_pred = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # get the inputs and labels
            inputs, labels = data['data'].to(device), data['label'].to(device)
            # forward
            outputs = model(inputs)
            # collect labels & prediction
            prediction = torch.max(outputs, 1)[1]
            all_label.extend(labels.squeeze())
            all_pred.extend(prediction)
    # Compute accuracy
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    all_label = all_label.squeeze().cpu().data.squeeze().numpy()
    all_pred = all_pred.cpu().data.squeeze().numpy()
    return all_label, all_pred


def plot_confusion_matrix(confmat, save_path='confmat.png', normalize=False):
    # Normalize the matrix
    if normalize:
        confmat = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
    # Draw matrix
    plt.figure(figsize=(20,20))
    # confmat = np.random.rand(100,100)
    plt.imshow(confmat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    # Add ticks
    ticks = np.arange(100)
    plt.xticks(ticks, fontsize=8)
    plt.yticks(ticks, fontsize=8)
    plt.grid(True)
    # Add title & labels
    plt.title('Confusion matrix', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.ylabel('True label', fontsize=20)
    # Save figure
    plt.savefig(save_path)


# Path setting
data_path = "/home/haodong/Data/CSL_Isolated/color_video_125000"
label_path = "/home/haodong/Data/CSL_Isolated/dictionary.txt"
model_path = "/home/haodong/Data/visualize_models/resnet18.pth"
# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]="3"
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
num_classes = 100
batch_size = 32
sample_size = 128
sample_duration = 16

# Load data
transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])
test_set = CSL_Isolated(data_path=data_path, label_path=label_path, frames=sample_duration,
    num_classes=num_classes, train=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Create model
model = resnet18(pretrained=True, progress=True, sample_size=sample_size, sample_duration=sample_duration, num_classes=num_classes).to(device)
# Run the model parallelly
if torch.cuda.device_count() > 1:
    logger.info("Using {} GPUs".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)
# Load model
model.load_state_dict(torch.load(model_path))

# Get prediction
all_label, all_pred = get_label_and_pred(model, test_loader, device)
matrix = confusion_matrix(all_label, all_pred)
plot_confusion_matrix(matrix)
sorted_index = np.diag(matrix).argsort()
for i in range(10):
    # print(sorted_index[i])
    print(test_set.label_to_word(int(sorted_index[i])))
