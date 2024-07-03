import os
from argparse import ArgumentParser
from sklearn.metrics import f1_score
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from data import CIFAR10Data
from module import CIFAR10Module

import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
import tarfile
import os

import torch, os
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')
from torchvision import transforms
from torch.utils import data
from PIL import Image
import pandas as pd
import torch.optim as optim
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, ConcatDataset
from cifar10_models.resnet import resnet18, resnet34, resnet50
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from cifar10_models.densenet import densenet121
from module import CIFAR10Module
import torchdrift
from sklearn import manifold
from matplotlib import pyplot
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels = np.load('CIFAR10/labels.npy')
top_labels = labels[:10000]
train_labels = np.load('CIFAR10/train_labels.npy')
test_labels = np.load('CIFAR10/test_labels.npy')
print(len(labels), len(top_labels))


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, labels, starting_index = 0, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            labels (numpy.array): NumPy array containing the labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_dir = img_dir
        self.labels = labels
        self.transform = transform
        image_files = []
        for i in range(starting_index, self.labels.shape[0] + starting_index):
            image_files.append(str(i) + '.png')


        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')  # Convert image to RGB
        
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Randomly crop the image with padding
    transforms.RandomHorizontalFlip(),     # Randomly flip the image horizontally
    transforms.ToTensor(),                 # Convert the image to a PyTorch tensor
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]),  # Normalize the image
    transforms.Lambda(lambda x: x.to(device))  # Move the tensor to the GPU if available
])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]),
    transforms.Lambda(lambda x: x.to(device))
])
train_dir = 'CIFAR10/CIFAR10/Train'
train_data = ImageDataset(train_dir, train_labels, starting_index=1, transform=transform_train)
train_loader = data.DataLoader(train_data, batch_size=32, shuffle=True)

test_dir = 'CIFAR10/Test'
test_data = ImageDataset(test_dir, top_labels, starting_index=1, transform=transform)
test_loader = data.DataLoader(test_data, batch_size=32, shuffle=False)

d1l1 = 'CIFAR10/D1/L1'
d1l1_data = ImageDataset(d1l1, top_labels, starting_index=0, transform=transform)
d1l1_loader = data.DataLoader(d1l1_data, batch_size=32, shuffle=False)

d1l2 = 'CIFAR10/D1/L2'
d1l2_data = ImageDataset(d1l2, top_labels, starting_index=0, transform=transform)
d1l2_loader = data.DataLoader(d1l2_data, batch_size=32, shuffle=False)

d2l1 = 'CIFAR10/D2/L1'
d2l1_data = ImageDataset(d2l1, top_labels, starting_index=0, transform=transform)
d2l1_loader = data.DataLoader(d2l1_data, batch_size=32, shuffle=False)

d2l2 = 'CIFAR10/D2/L2'
d2l2_data = ImageDataset(d2l2, top_labels, starting_index=0, transform=transform)
d2l2_loader = data.DataLoader(d2l2_data, batch_size=32, shuffle=False)

d3l1 = 'CIFAR10/D3/L1'
d3l1_data = ImageDataset(d3l1, top_labels, starting_index=0, transform=transform)
d3l1_loader = data.DataLoader(d3l1_data, batch_size=32, shuffle=False)

d3l2 = 'CIFAR10/D3/L2'
d3l2_data = ImageDataset(d3l2, top_labels, starting_index=0, transform=transform)
d3l2_loader = data.DataLoader(d3l2_data, batch_size=32, shuffle=False)

d4l1 = 'CIFAR10/D4/L1'
d4l1_data = ImageDataset(d4l1, top_labels, starting_index=0, transform=transform)
d4l1_loader = data.DataLoader(d4l1_data, batch_size=32, shuffle=False)

d4l2 = 'CIFAR10/D4/L2'
d4l2_data = ImageDataset(d4l2, top_labels, starting_index=0, transform=transform)
d4l2_loader = data.DataLoader(d4l2_data, batch_size=32, shuffle=False)

d1 = ConcatDataset([d1l1_data, d1l2_data])
d1_loader = data.DataLoader(d1,batch_size=32, shuffle=False )

d2 = ConcatDataset([d2l1_data, d2l2_data])
d2_loader = data.DataLoader(d2, batch_size=32, shuffle=False )

d3 = ConcatDataset([d3l1_data, d3l2_data])
d3_loader = data.DataLoader(d3, batch_size=32, shuffle=False )

d4 = ConcatDataset([d4l1_data, d4l2_data])
d4_loader = data.DataLoader(d4, batch_size=32, shuffle=False )
# Pretrained model
my_model = densenet121(pretrained=True)
my_model.fc = torch.nn.Identity()
my_model = my_model.to(device)


def retrain_model(model, train_loader, num_epochs=15):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')
    return model



from sklearn.metrics import accuracy_score

def testing(test_loader, my_model, dataset):
    my_model.eval() 

    test_correct = 0
    pred_label = []
    true_label = []
    N = len(test_loader.dataset)
    with torch.no_grad():
        for x, y in test_loader:
            x, y=x.to(device), y.to(device)
            z = my_model(x)
            _, yhat = torch.max(z.data, 1)
            true_label.append(y)
            pred_label.append(yhat)
            test_correct += (yhat == y).sum().item()
    test_acc = test_correct / N
    
    pred_arr = np.concatenate([tensor.cpu().numpy() for tensor in pred_label])
    true_arr = np.concatenate([tensor.cpu().numpy() for tensor in true_label])
    f1 = f1_score(true_arr, pred_arr, average='macro')
    print(f"{dataset} Accuracy: {test_acc}, Macro F1: {f1}")
    # return pred_label

def drift_detect(data_loader, my_model, data_name):
    inputs, _ = next(iter(data_loader))
    drift_detector = torchdrift.detectors.KernelMMDDriftDetector()
    torchdrift.utils.fit(test_loader, my_model, drift_detector)
    drift_detection_model = torch.nn.Sequential(
        my_model,
        drift_detector
    )

    features = my_model(inputs)
    score = drift_detector(features)
    p_val = drift_detector.compute_p_value(features)
    print(f'{data_name} score: {score}, p_value: {p_val}')

    base_outputs_numpy = drift_detector.base_outputs.cpu().detach().numpy()
    mapper = manifold.Isomap(n_components=2)
    base_embedded = mapper.fit_transform(base_outputs_numpy)
    features = features.cpu().detach().numpy()
    features_embedded = mapper.transform(features)
    pyplot.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c='r')
    pyplot.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
    pyplot.title(f'{data_name}: score {score:.2f} p-value {p_val:.2f}')
    pyplot.show()


# retrained_model = retrain_model(my_model, train_loader, num_epochs=15)
# testing(d1_loader,'D1')
# testing(d2_loader,'D2')
# testing(d3_loader,'D3')
# testing(d4_loader,'D4')

# testing(test_loader, my_model, 'test set')
# testing(d1l1_loader,my_model,'D1 L1')
# testing(d1l2_loader,my_model,'D1 L2')
# testing(d2l1_loader, my_model, 'D2 L1')
# testing(d2l1_loader, retrained_model, 'D2 L1')
# testing(d2l2_loader, my_model, 'D2 L2')
# testing(d3l1_loader, my_model,'D3 L1')
# testing(d3l2_loader, my_model, 'D3 L2')
# testing(d4l1_loader, my_model, 'D4 L1')
# testing(d4l2_loader, my_model, 'D4 L2')
# drift_detect(d1_loader, my_model, 'D1')
# drift_detect(d2_loader, my_model, 'D2')
# drift_detect(d3_loader, my_model, 'D3')
# drift_detect(d4_loader, my_model, 'D4')
# drift_detect(test_loader, my_model, 'test set')
# drift_detect(d1l1_loader, my_model, 'D1 L1')
# drift_detect(d1l2_loader, my_model, 'D1 L2')
# drift_detect(d2l1_loader, my_model, 'D2 L1')
# drift_detect(d2l2_loader, my_model, 'D2 L2')
# drift_detect(d3l1_loader, my_model, 'D3 L1')
# drift_detect(d3l2_loader, my_model, 'D3 L2')
# drift_detect(d4l1_loader, my_model, 'D4 L1')
# drift_detect(d4l2_loader, my_model, 'D4 L2')
