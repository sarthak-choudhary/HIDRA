# BSD 2-Clause License

# Copyright (c) 2022, Lun Wang
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
    Generate general data, malicious data and heterogenous data.
'''

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms
from PIL import Image
import os

MAL_FEATURE_TEMPLATE = '../data/%s_mal_feature_10.npy'
MAL_TARGET_TEMPLATE = '../data/%s_mal_target_10.npy'
MAL_TRUE_LABEL_TEMPLATE = '../data/%s_mal_true_label_10.npy'

class PurchaseDataset(Dataset):
    def __init__(self, filename):
        self.features =  []
        self.labels = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                splitted_line = line.split(",")
                self.labels.append(int(splitted_line[0]) - 1)
                self.features.append(list(map(int, splitted_line[1:])))

        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.int64)
        self.transform = transforms.Compose([transforms.ToTensor()])    

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]  

class MITIndoorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_paths = []
        self.labels = []

        # Load images paths and labels

        for label, class_dir in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for image_file in os.listdir(class_path):
                    self.images_paths.append(os.path.join(class_path, image_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images_paths)
        
    def __getitem__(self, idx):
        image = Image.open(self.images_paths[idx])

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize((224, 224), Image.ANTIALIAS)

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, label
        
class MalDataset(Dataset):
    def __init__(self, feature_path, true_label_path, target_path, transform=None):
        self.feature = np.load(feature_path)
        self.mal_dada = np.load(feature_path)
        self.true_label = np.load(true_label_path)
        self.target = np.load(target_path)

        self.transform = transform
    
    def __getitem__(self, idx):
        sample = self.feature[idx]
        mal_data = self.mal_dada[idx]
        if self.transform:
            sample = self.transform(sample)
            mal_data = self.transform(mal_data)
        return sample, mal_data, self.true_label[idx], self.target[idx]
    
    def __len__(self):
        return self.target.shape[0]

def gen_mal_mnist(batch_size=10):
    torch.manual_seed(hash('mnist'))
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()])
    test_set = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    sizes = [batch_size] * (len(test_set) // batch_size)
    test_sets = random_split(test_set, sizes)
    for idx, (feature, target) in enumerate(DataLoader(test_sets[0], batch_size=10, shuffle=True), 0):
        np.save(MAL_FEATURE_TEMPLATE%('mnist'), feature.numpy().transpose([0,3,2,1]))
        np.save(MAL_TRUE_LABEL_TEMPLATE%('mnist'), target.numpy())
        mal_train_labels = target.numpy().copy()
        for i in range(target.shape[0]):
            allowed_targets = list(range(10))
            allowed_targets.remove(target[i])
            mal_train_labels[i] = np.random.choice(allowed_targets)
        np.save(MAL_TARGET_TEMPLATE%('mnist'), mal_train_labels)

def gen_mal_fashion(batch_size=10):
    torch.manual_seed(hash('fashion'))
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()])
    test_set = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)
    sizes = [batch_size] * (len(test_set) // batch_size)
    test_sets = random_split(test_set, sizes)
    for idx, (feature, target) in enumerate(DataLoader(test_sets[0], batch_size=10, shuffle=True), 0):
        np.save(MAL_FEATURE_TEMPLATE%('fashion'), feature.numpy().transpose([0,3,2,1]))
        np.save(MAL_TRUE_LABEL_TEMPLATE%('fashion'), target.numpy())
        mal_train_labels = target.numpy().copy()
        for i in range(target.shape[0]):
            allowed_targets = list(range(10))
            allowed_targets.remove(target[i])
            mal_train_labels[i] = np.random.choice(allowed_targets)
        np.save(MAL_TARGET_TEMPLATE%('fashion'), mal_train_labels)

if __name__ == '__main__':    
    gen_mal_mnist()
    gen_mal_fashion()
