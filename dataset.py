import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

"""
Implementation of Chinese Sign Language Dataset(50 signers with 5 times)
"""
class CSL_Isolated(Dataset):
    def __init__(self, data_path, label_path, frames=16, num_classes=500, train=True, transform=None):
        super(CSL_Isolated, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.train = train
        self.transform = transform
        self.frames = frames
        self.num_classes = num_classes
        self.signers = 50
        self.repetition = 5
        if self.train:
            self.videos_per_folder = int(0.8 * self.signers * self.repetition)
        else:
            self.videos_per_folder = int(0.2 * self.signers * self.repetition)
        self.data_folder = []
        try:
            obs_path = [os.path.join(self.data_path, item) for item in os.listdir(self.data_path)]
            self.data_folder = sorted([item for item in obs_path if os.path.isdir(item)])
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
        except Exception as e:
            raise

    def read_images(self, folder_path):
        assert len(os.listdir(folder_path)) >= self.frames, "Too few images in your data folder: " + str(folder_path)
        images = []
        start = 1
        step = int(len(os.listdir(folder_path))/self.frames)
        for i in range(self.frames):
            image = Image.open(os.path.join(folder_path, '{:06d}.jpg').format(start+i*step))  #.convert('L')
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images, dim=0)
        # switch dimension for 3d cnn
        images = images.permute(1, 0, 2, 3)
        # print(images.shape)
        return images

    def __len__(self):
        return self.num_classes * self.videos_per_folder

    def __getitem__(self, idx):
        top_folder = self.data_folder[int(idx/self.videos_per_folder)]
        selected_folders = [os.path.join(top_folder, item) for item in os.listdir(top_folder)]
        selected_folders = sorted([item for item in selected_folders if os.path.isdir(item)])
        if self.train:
            selected_folder = selected_folders[idx%self.videos_per_folder]
        else:
            selected_folder = selected_folders[idx%self.videos_per_folder + int(0.8*self.signers*self.repetition)]
        images = self.read_images(selected_folder)
        # print(selected_folder, int(idx/self.videos_per_folder))
        # print(self.labels['{:06d}'.format(int(idx/self.videos_per_folder))])
        # label = self.labels['{:06d}'.format(int(idx/self.videos_per_folder))]
        label = torch.LongTensor([int(idx/self.videos_per_folder)])

        return {'images': images, 'label': label}

    def label_to_word(self, label):
        return self.labels['{:06d}'.format(label.item())]


"""
Implementation of Chinese Sign Language Dataset(50 signers with 1 time)
"""
class CSL_Isolated_25000(Dataset):
    def __init__(self, data_path, label_path, frames=16, videos_per_folder=250, transform=None):
        super(CSL_Isolated_25000, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.frames = frames
        self.data_folder = []
        self.videos_per_folder = videos_per_folder # 50 for CSL_Isolated_25000, 250 for CSL_Isolated_125000
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
        assert len(os.listdir(folder_path)) >= self.frames, "Too few images in your data folder: " + str(folder_path)
        images = []
        # ignore the first image
        start = 2
        step = int(((len(os.listdir(folder_path))-1))/self.frames)
        for i in range(self.frames):
            image = Image.open(os.path.join(folder_path, '{:06d}.jpg').format(start+i*step)).convert('L')
            # crop the image using Pillow
            image = image.crop([384, 240, 896, 720])
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images, dim=0)
        # switch dimension for 3d cnn
        images = images.permute(1, 0, 2, 3)
        # print(images.shape)
        return images

    def __len__(self):
        # return 100 * self.videos_per_folder
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

"""
Implementation of CSL Skeleton Dataset
"""
class CSL_Skeleton(Dataset):
    joints_index = {'SPINEBASE': 0, 'SPINEMID': 1, 'NECK': 2, 'HEAD': 3, 'SHOULDERLEFT':4,
                    'ELBOWLEFT': 5, 'WRISTLEFT': 6, 'HANDLEFT': 7, 'SHOULDERRIGHT': 8,
                    'ELBOWRIGHT': 9, 'WRISTRIGHT': 10, 'HANDRIGHT': 11, 'HIPLEFT': 12,
                    'KNEELEFT': 13, 'ANKLELEFT': 14, 'FOOTLEFT': 15, 'HIPRIGHT': 16,
                    'KNEERIGHT': 17, 'ANKLERIGHT': 18, 'FOOTRIGHT': 19, 'SPINESHOULDER': 20,
                    'HANDTIPLEFT': 21, 'THUMBLEFT': 22, 'HANDTIPRIGHT': 23, 'THUMBRIGHT': 24}
    def __init__(self, data_path, label_path, frames=16, num_classes=500, selected_joints=None, split_to_channels=False, train=True, transform=None):
        super(CSL_Skeleton, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.frames = frames
        self.num_classes = num_classes
        self.selected_joints = selected_joints
        self.split_to_channels = split_to_channels
        self.train = train
        self.transform = transform
        self.signers = 50
        self.repetition = 5
        if self.train:
            self.txt_per_folder = int(0.8 * self.signers * self.repetition)
        else:
            self.txt_per_folder = int(0.2 * self.signers * self.repetition)
        self.data_folder = []
        try:
            obs_path = [os.path.join(self.data_path, item) for item in os.listdir(self.data_path)]
            self.data_folder = sorted([item for item in obs_path if os.path.isdir(item)])
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
        except Exception as e:
            raise

    def read_file(self, txt_path):
        txt_file = open(txt_path, 'r')
        all_skeletons = []
        for line in txt_file.readlines():
            line = line.split(' ')
            skeleton = [int(item) for item in line if item is not '\n']
            selected_x = []
            selected_y = []
            # select specific joints
            if self.selected_joints is not None:
                for joint in self.selected_joints:
                    assert joint in self.joints_index, 'JOINT ' + joint + ' DONT EXIST!!!'
                    selected_x.append(skeleton[2*self.joints_index[joint]])
                    selected_y.append(skeleton[2*self.joints_index[joint]+1])
            else:
                for i in range(len(skeleton)):
                    if i % 2 == 0:
                        selected_x.append(skeleton[i])
                    else:
                        selected_y.append(skeleton[i])
            # print(selected_x, selected_y)
            if self.split_to_channels:
                selected_skeleton = torch.FloatTensor([selected_x, selected_y])
            else:
                selected_skeleton = torch.FloatTensor(selected_x + selected_y)
            # print(selected_skeleton.shape)
            if self.transform is not None:
                selected_skeleton = self.transform(selected_skeleton)
            all_skeletons.append(selected_skeleton)
        # print(all_skeletons)
        skeletons = []
        start = 0
        step = int(len(all_skeletons)/self.frames)
        for i in range(self.frames):
            skeletons.append(all_skeletons[start+i*step])
        skeletons = torch.stack(skeletons, dim=0)
        # print(skeletons.shape)

        return skeletons

    def __len__(self):
        return self.num_classes * self.txt_per_folder

    def __getitem__(self, idx):
        top_folder = self.data_folder[int(idx/self.txt_per_folder)]
        selected_txts = [os.path.join(top_folder, item) for item in os.listdir(top_folder)]
        selected_txts = sorted([item for item in selected_txts if item.endswith('.txt')])
        if self.train:
            selected_txt = selected_txts[idx%self.txt_per_folder]
        else:
            selected_txt = selected_txts[idx%self.txt_per_folder + int(0.8*self.signers*self.repetition)]
        # print(selected_txt)
        data = self.read_file(selected_txt)
        label = torch.LongTensor([int(idx/self.txt_per_folder)])

        return {'images': data, 'label': label}

    def label_to_word(self, label):
        return self.labels['{:06d}'.format(label.item())]


# Test
if __name__ == '__main__':
    # transform = transforms.Compose([transforms.Resize([128, 128]), transforms.ToTensor()])
    # dataset = CSL_Isolated(data_path="/home/aistudio/data/data20273/CSL_Isolated_125000",
    #     label_path='/home/aistudio/data/data20273/CSL_Isolated_125000/dictionary.txt', transform=transform)    # print(len(dataset))
    # # print(dataset[1000]['images'].shape)
    dataset = CSL_Skeleton(data_path="/home/haodong/Data/CSL_Isolated_1/xf500_body_depth_txt",
        label_path="/home/haodong/Data/CSL_Isolated_1/dictionary.txt", selected_joints=['SPINEBASE', 'SPINEMID', 'HANDTIPRIGHT'], split_to_channels=True)
    # print(dataset[1000])
    # label = dataset[1000]['label']
    # print(dataset.label_to_word(label))
    dataset[1000]
