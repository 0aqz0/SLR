import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

"""
Implementation of Chinese Sign Language Dataset
"""
class CSL_Dataset(Dataset):
    def __init__(self, data_path, label_path, frames=30, transform=None):
        super(CSL_Dataset, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.frames = frames
        self.data_folder = []
        self.videos_per_folder = 50
        try:
            obs_path = [os.path.join(self.data_path, item) for item in os.listdir(self.data_path)]
            self.data_folder = [item for item in obs_path if os.path.isdir(item)]
            self.data_folder.sort()
        except Exception as e:
            print("Something wrong with your data path!!!")
            raise
        self.labels = {}
        try:
            label_file = open(self.label_path, 'r')
            for line in label_file.readlines():
                line = line.strip()
                line = line.split('\t')
                self.labels[line[0]] = line[1]
            # print(self.labels['000001'])
        except Exception as e:
            raise

    def read_images(self, folder_path):
        assert len(os.listdir(folder_path)) >= self.frames, "Too few images in your data folder!!!"
        images = []
        # ignore the first image
        start = 2
        step = int(((len(os.listdir(folder_path))-1))/self.frames)
        for i in range(self.frames):
            image = Image.open(os.path.join(folder_path, '{:06d}.jpg').format(start+step*i)).convert('L')
            if self.transform is not None:
                image = self.transform(image)
            # print(image.squeeze(0).shape)
            images.append(image.squeeze(0))
        images = torch.stack(images, dim=0)
        # print(images.shape)
        return images

    def __len__(self):
        return len(self.data_folder) * self.videos_per_folder

    def __getitem__(self, idx):
        top_folder = self.data_folder[int(idx/self.videos_per_folder)]
        selected_folders = [os.path.join(top_folder, item) for item in os.listdir(top_folder)]
        selected_folders = [item for item in selected_folders if os.path.isdir(item)]
        selected_folder = selected_folders[idx%self.videos_per_folder]
        # print(selected_folder)
        images = self.read_images(selected_folder)
        # print(self.labels['{:06d}'.format(int(idx/self.videos_per_folder))])
        # label = self.labels['{:06d}'.format(int(idx/self.videos_per_folder))]
        label = torch.LongTensor([int(idx/self.videos_per_folder)])

        return {'images': images, 'label': label}

    def label_to_word(self, label):
        return self.labels['{:06d}'.format(label.item())]


# Test
if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize([90, 120]), transforms.ToTensor()])
    dataset = CSL_Dataset(data_path="/media/zjunlict/TOSHIBA1/dataset/SLR_dataset/S500_color_video", 
        label_path='/media/zjunlict/TOSHIBA1/dataset/SLR_dataset/dictionary.txt', transform=transform)
    print(len(dataset))
    print(dataset[1000])
    label = dataset[1000]['label']
    print(dataset.label_to_word(label))

