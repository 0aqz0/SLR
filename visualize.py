import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, confusion_matrix
from dataset import CSL_Isolated
from models.Conv3D import resnet18, resnet34, resnet50, r2plus1d_18
import numpy as np
import matplotlib.pyplot as plt
from numpy import savetxt
import os
import argparse

def get_label_and_pred(model, dataloader, device):
    all_label = []
    all_pred = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # get the inputs and labels
            inputs, labels = data['data'].to(device), data['label'].to(device)
            # forward
            outputs = model(inputs)
            if isinstance(outputs, list):
                outputs = outputs[0]
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


def plot_confusion_matrix(model, dataloader, device, save_path='confmat.png', normalize=True):
    # Get prediction
    all_label, all_pred = get_label_and_pred(model, dataloader, device)
    confmat = confusion_matrix(all_label, all_pred)

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

    # Ranking
    sorted_index = np.diag(confmat).argsort()
    for i in range(10):
        # print(type(sorted_index[i]))
        print(test_set.label_to_word(int(sorted_index[i])), confmat[sorted_index[i]][sorted_index[i]])
    # Save to csv
    savetxt('matrix.csv', confmat, delimiter=',')


# Parameters manager
parser = argparse.ArgumentParser(description='Visualization')
parser.add_argument('--data_path', default='/home/haodong/Data/CSL_Isolated/color_video_125000',
    type=str, help='Path to data')
parser.add_argument('--label_path', default='/home/haodong/Data/CSL_Isolated/dictionary.txt',
    type=str, help='Path to labels')
parser.add_argument('--model', default='resnet18',
    type=str, help='Model to use')
parser.add_argument('--checkpoint', default='/home/haodong/Data/visualize_models/resnet18.pth',
    type=str, help='Path to checkpoint')
parser.add_argument('--device', default='0',
    type=str, help='CUDA visible devices')
parser.add_argument('--num_classes', default=100,
    type=int, help='Number of classes')
parser.add_argument('--batch_size', default=16,
    type=int, help='Batch size')
parser.add_argument('--sample_size', default=128,
    type=int, help='Sample size')
parser.add_argument('--sample_duration', default=16,
    type=int, help='Sample duration')
parser.add_argument('--confusion_matrix', action='store_true',
    help='Draw confusion matrix')
args = parser.parse_args()

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
num_classes = args.num_classes
batch_size = args.batch_size
sample_size = args.sample_size
sample_duration = args.sample_duration

if __name__ == '__main__':
    # Load data
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    test_set = CSL_Isolated(data_path=args.data_path, label_path=args.label_path, frames=sample_duration,
        num_classes=num_classes, train=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # Create model
    if args.model == 'resnet18':
        model = resnet18(pretrained=True, progress=True, sample_size=sample_size,
            sample_duration=sample_duration, num_classes=num_classes).to(device)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Load model
    model.load_state_dict(torch.load(args.checkpoint))

    # Draw confusion matrix
    if args.confusion_matrix:
        plot_confusion_matrix(model, test_loader, device)
