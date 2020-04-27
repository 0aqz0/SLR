import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as utils
import cv2
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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
    np.savetxt('matrix.csv', confmat, delimiter=',')


def visualize_attn(I, c):
    # Image
    img = I.permute((1,2,0)).cpu().numpy()
    # Heatmap
    N, C, H, W = c.size()
    a = F.softmax(c.view(N,C,-1), dim=2).view(N,C,H,W)
    up_factor = 128/H
    # print(up_factor, I.size(), c.size())
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=4, normalize=True, scale_each=True)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    # Add the heatmap to the image
    vis = 0.6 * img + 0.4 * attn
    return torch.from_numpy(vis).permute(2,0,1)


def plot_attention_map(model, dataloader, device):
    # Summary writer
    writer = SummaryWriter("runs/attention_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now()))

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # get images
            inputs = data['data'].to(device)
            if batch_idx == 0:
                images = inputs[0:16,:,:,:,:]
                I = utils.make_grid(images[:,:,0,:,:], nrow=4, normalize=True, scale_each=True)
                writer.add_image('origin', I)
                _, c1, c2, c3, c4 = model(images)
                # print(I.shape, c1.shape, c2.shape, c3.shape, c4.shape)
                attn1 = visualize_attn(I, c1[:,:,0,:,:])
                writer.add_image('attn1', attn1)
                attn2 = visualize_attn(I, c2[:,:,0,:,:])
                writer.add_image('attn2', attn2)
                attn3 = visualize_attn(I, c3[:,:,0,:,:])
                writer.add_image('attn3', attn3)
                attn4 = visualize_attn(I, c4[:,:,0,:,:])
                writer.add_image('attn4', attn4)
                break


"""
Calculate Word Error Rate
Word Error Rate = (Substitutions + Insertions + Deletions) / Number of Words Spoken
Reference:
https://holianh.github.io/portfolio/Cach-tinh-WER/
https://github.com/imalic3/python-word-error-rate
"""
def wer(r, h):
    # initialisation
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return float(d[len(r)][len(h)]) / len(r) * 100


if __name__ == '__main__':
    # Calculate WER
    r = [1,2,3,4]
    h = [1,1,3,5,6]
    print(wer(r, h))
