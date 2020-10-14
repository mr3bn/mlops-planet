import os
from PIL import Image

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

IMG_PATH = '../data/train-tif-v2/'
IMG_EXT = '.tif'
TRAIN_DATA = '../data/train_v2.csv/train_v2.csv'
NUM_EPOCHS = 2

class KaggleAmazonDataset(Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, csv_path, img_path, img_ext, transform=None):
    
        tmp_df = pd.read_csv(csv_path)
        assert tmp_df['image_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
"Some images referenced in the CSV file were not found"
        
        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df['image_name']
        self.y_train = self.mlb.fit_transform(tmp_df['tags'].str.split()).astype(np.float32)

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        label = torch.from_numpy(self.y_train[index])
        return img, label

    def __len__(self):
        return len(self.X_train.index)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2304, 256)
        self.fc2 = nn.Linear(256, 17)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1) # Flatten layer
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.sigmoid(x)

def train_one_epoch(model, optimizer, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()

def prep_data_loader(train_data, img_path, img_ext, transformations):
    train_labels = pd.read_csv(train_data)


    dset_train = KaggleAmazonDataset(TRAIN_DATA,IMG_PATH,IMG_EXT,transformations)

    train_loader = DataLoader(dset_train,
                          batch_size=256,
                          shuffle=True,
                          num_workers=4 # 1 for CUDA
                         # pin_memory=True # CUDA only
                         )

    return train_loader


def train_model(model, num_epochs, optimizer, train_loader):

    model.train()

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader)

    return model

if __name__ == '__main__':

    transformations = transforms.Compose([transforms.Scale(32), transforms.ToTensor()])

    model = Net()
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    train_data_loader = prep_data_loader(TRAIN_DATA, IMG_PATH, IMG_EXT, transformations)

    train_model(model, NUM_EPOCHS, optimizer, train_data_loader)