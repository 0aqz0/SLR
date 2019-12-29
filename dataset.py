import os
from PIL import Image
from torch.utils.data import Dataset

class SLR_Dataset(Dataset):
    def __init__(self, data_path, label_path, frames=50, transform=None):
        super(SLR_Dataset, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.frames = frames
        self.data_folder = []
        try:
            obs_path = [os.path.join(self.data_path, item) for item in os.listdir(self.data_path)]
            self.data_folder = [item for item in obs_path if os.path.isdir(item)]
        except Exception as e:
            print("Something wrong with your data path!!!")
            raise
        # try:
        #     pass
        # except Exception as e:
        #     raise
    def read_images(self, folder_path):
        assert len(os.listdir(folder_path)) >= self.frames, "Too few images in your data folder!!!"
        images = []
        for i in range(self.frames):
            image = Image.open(os.path.join(folder_path, '{:06d}.jpg').format(i))
            images.append(image)

        return images

    def __len__(self):
        return len(self.data_folder)

    def __getitem__(self, idx):
        selected_folder = self.data_folder[idx]
        images = self.read_images(selected_folder)


if __name__ == '__main__':
    dataset = SLR_Dataset(".", '')
    print(len(dataset))
    print(dataset[0])

