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
    The simulation program for Federated Learning.
'''

import argparse
from attack import attack_krum, attack_trimmedmean, attack_xie, backdoor, mal_single, corrupt_grads, attack_single_direction, partial_attack_single_direction
from data import MalDataset, PurchaseDataset, MITIndoorDataset
from networks import ConvNet, EMNISTCNN, FCs
import numpy as np
import random
from robust_estimator import krum, filterL2, median, trimmed_mean, bulyan, ex_noregret, mom_filterL2, mom_ex_noregret, mom_krum
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import utils as vutils
from torchvision import datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from emnist import extract_training_samples
import time
import copy
import os

random.seed(2022)
np.random.seed(5)
torch.manual_seed(11)
torch.set_default_dtype(torch.float64)

MAL_FEATURE_FILE = './data/%s_mal_feature_10.npy'
MAL_TARGET_FILE = './data/%s_mal_target_10.npy'
MAL_TRUE_LABEL_FILE = './data/%s_mal_true_label_10.npy'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1')
    parser.add_argument('--dataset', default='MNIST')
    parser.add_argument('--nworker', type=int, default=100)
    parser.add_argument('--perround', type=int, default=100)
    parser.add_argument('--localiter', type=int, default=5)
    parser.add_argument('--round', type=int, default=100) 
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--checkpoint', type=int, default=1)
    parser.add_argument('--sigma', type=float, default=1e-5)
    parser.add_argument('--batchsize', type=int, default=50)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--tau', type=float, default=10.)
    parser.add_argument('--buckets', type=int, default=10)
    parser.add_argument('--adaptive_threshold', action='store_true', help="Enabale adaptive threshold")

    # Malicious agent setting
    parser.add_argument('--malnum', type=int, default=20)
    parser.add_argument('--agg', default='average', help='average, ex_noregret, filterl2, krum, median, trimmedmean, bulyankrum, bulyantrimmedmean, bulyanmedian, mom_filterl2, mom_ex_noregret, iclr2022_bucketing, icml2021_history, clustering')
    parser.add_argument('--attack', default='noattack', help="noattack, trimmedmean, krum, backdoor, modelpoisoning, xie")
    args = parser.parse_args()

    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu") 
    mal_index = list(range(args.malnum))
    benign_index = list(range(args.malnum, args.nworker))

    if args.dataset == 'MNIST':
        transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))])

        train_set = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
        # batch_size = len(train_set) // args.nworker
        train_loader = DataLoader(train_set, batch_size=args.batchsize)
        test_loader = DataLoader(torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform))
        # mal_train_loaders = DataLoader(MalDataset(MAL_FEATURE_FILE%('mnist'), MAL_TRUE_LABEL_FILE%('mnist'), MAL_TARGET_FILE%('mnist'), transform=transform), batch_size=args.batchsize)
        mal_train_loaders = None

        network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)
        backdoor_network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)
    elif args.dataset == 'Fashion-MNIST':
        train_set = torchvision.datasets.FashionMNIST(root = "./data", train = True, download = True, transform = torchvision.transforms.ToTensor())
        # batch_size = len(train_set) // args.nworker
        train_loader = DataLoader(train_set, batch_size=args.batchsize)
        test_loader = DataLoader(torchvision.datasets.FashionMNIST(root = "./data", train = False, download = True, transform = torchvision.transforms.ToTensor()))
        # mal_train_loaders = DataLoader(MalDataset(MAL_FEATURE_FILE%('fashion'), MAL_TRUE_LABEL_FILE%('fashion'), MAL_TARGET_FILE%('fashion'), transform=torchvision.transforms.ToTensor()), batch_size=args.batchsize)

        mal_train_loaders = None
        network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)
        backdoor_network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)
    elif args.dataset == 'EMNIST':
        images, labels = extract_training_samples('digits')
        train_images = torch.tensor((images[:200000]/255-0.5).reshape(200000, 1, 28, 28))
        train_labels = labels[:200000]
        valid_images = torch.tensor((images[200000:]/255-0.5).reshape(40000, 1, 28, 28))
        valid_labels = labels[200000:]
        train_set = list(zip(train_images.float(), train_labels.astype('int64')))
        test_set = list(zip(valid_images.float(), valid_labels.astype('int64')))
        test_loader = DataLoader(test_set)

        network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)
        backdoor_network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)
    elif args.dataset == 'CIFAR10':
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        transform = transforms.Compose([
            transforms.ToTensor(),
            ])
        
        train_set = datasets.CIFAR10('./data', train=True,
                              download=True, transform=transform_train)
        test_set = datasets.CIFAR10('./data', train=False,
                             download=True, transform=transform)
        train_loader = DataLoader(train_set, batch_size=args.batchsize)
        test_loader = DataLoader(test_set)

        # mal_train_loaders = DataLoader(MalDataset(MAL_FEATURE_FILE%('cifar'), MAL_TRUE_LABEL_FILE%('cidar'), MAL_TARGET_FILE%('cifar'), transform=torchvision.transforms.ToTensor()), batch_size=args.batchsize)
        mal_train_loaders = None
        network = ConvNet(input_size=32, input_channel=3, classes=10, kernel_size=5, filters1=64, filters2=64, fc_size=384).to(device)
        backdoor_network = ConvNet(input_size=32, input_channel=3, classes=10, kernel_size=5, filters1=64, filters2=64, fc_size=384).to(device)
    elif args.dataset == 'Purchase':
        filename = './data/dataset_purchase'
        dataset = PurchaseDataset(filename)
        train_set = list(zip(dataset.features[:-10000].float(), dataset.labels[:-10000]))
        test_set = list(zip(dataset.features[-10000:].float(), dataset.labels[-10000:]))

        train_loader = DataLoader(train_set, batch_size=args.batchsize)
        test_loader = DataLoader(test_set)

        mal_train_loaders = None
        network = FCs(in_ch=600, h_ch=512, out_ch=100).to(device)
        backdoor_network = FCs(in_ch=600, h_ch=512, out_ch=100).to(device)
    elif args.dataset == "SVHN":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        test_set = datasets.SVHN(root='./data', split='test', download=True, transform=transform)

        train_loader = DataLoader(train_set, batch_size=args.batchsize)
        test_loader = DataLoader(test_set)

        mal_train_loaders = None
        network = ConvNet(input_size=32, input_channel=3, classes=10, kernel_size=5, filters1=64, filters2=64, fc_size=384).to(device)
        backdoor_network = ConvNet(input_size=32, input_channel=3, classes=10, kernel_size=5, filters1=64, filters2=64, fc_size=384).to(device)
    elif args.dataset == "Food101":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pre-trained models
        ])

        train_set = datasets.Food101(root='./data', split="train", transform=transform, download=True)
        test_set = datasets.Food101(root='./data', split="test", transform=transform, download=True)
        
        train_loader = DataLoader(train_set, batch_size=args.batchsize)
        test_loader = DataLoader(test_set)

        mal_train_loaders = None
        network = ConvNet(input_size=224, input_channel=3, classes=101, kernel_size=3, filters1=64, filters2=64, fc_size=256).to(device)
        backdoor_network = ConvNet(input_size=224, input_channel=3, classes=101, kernel_size=3, filters1=64, filters2=64, fc_size=256).to(device)
    elif args.dataset == "PCAM":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        transform = transforms.Compose([
            transforms.ToTensor(),
            ])
        
        train_set = datasets.PCAM(root='./data', split="train", transform=transform_train, download=False)
        test_set = datasets.PCAM(root='./data', split="test", transform=transform, download=False)

        train_loader = DataLoader(train_set, batch_size=len(train_set) // args.nworker)
        test_loader = DataLoader(test_set)

        mal_train_loaders = None

        network = ConvNet(input_size=32, input_channel=3, classes=2, kernel_size=5, filters1=32, filters2=64, fc_size=128).to(device)
        backdoor_network = ConvNet(input_size=32, input_channel=3, classes=2, kernel_size=5, filters1=32, filters2=64, fc_size=128).to(device)
        # network = GCNN().to(device)
        # backdoor_network = GCNN().to(device)
    elif args.dataset == "MIT67":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        transform = transforms.Compose([
            transforms.ToTensor(),
            ])
        dataset = MITIndoorDataset(root_dir='./data/MIT67', transform=transform)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_set, batch_size=args.batchsize)
        test_loader = DataLoader(test_set)

        mal_train_loaders = None
        network = ConvNet(input_size=32, input_channel=3, classes=67, kernel_size=5, filters1=64, filters2=64, fc_size=256).to(device)
        backdoor_network = ConvNet(input_size=32, input_channel=3, classes=67, kernel_size=5, filters1=64, filters2=64, fc_size=256).to(device)
    # Split into multiple training set
    local_size = len(train_set) // args.nworker
    sizes = []
    sum = 0
    for i in range(0, args.nworker):
        sizes.append(local_size)
        sum = sum + local_size

    for i in range(0, len(train_set) - sum):
        sizes[i] = sizes[i] + 1

    train_sets = random_split(train_set, sizes)
    train_loaders = []
    for trainset in train_sets:
        train_loaders.append(DataLoader(trainset, batch_size=args.batchsize, shuffle=True))

    # define training loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=args.lr)
    # prepare data structures to store local gradients
    local_grads = []
    for i in range(args.nworker):
        local_grads.append([])
        for p in list(network.parameters()):
            local_grads[i].append(np.zeros(p.data.shape))
    prev_average_grad = None

    print('Malicious node indices:', mal_index, 'Attack Type:', args.attack)
    
    if not os.path.exists('./results/'):
        os.makedirs('./results/')

    if not os.path.exists('./variance_checkpoints/'):
        os.makedirs('./variance_checkpoints/')

    if not os.path.exists('./outlier_removal_checkpoints/'):
        os.makedirs('./outlier_removal_checkpoints/')

    file_name = './results/' + args.attack + '_' + args.agg + '_' + args.dataset + '_' + str(args.malnum) + '_' + str(args.lr) + '_' + str(args.localiter) + ('_adap_thres' if args.adaptive_threshold else '') + '.txt'
    checkpoint_file_name = './variance_checkpoints/' + args.attack + '_' + args.agg + '_' + args.dataset + '_' + str(args.malnum) + '_' + str(args.lr) + '_' + str(args.localiter) + ('_adap_thres' if args.adaptive_threshold else '') + '.txt'
    outlier_file_name = './outlier_removal_checkpoints/' +  args.attack + '_' + args.agg + '_' + args.dataset + '_' + str(args.malnum) + '_' + str(args.lr) + '_' + str(args.localiter) + ('_adap_thres' if args.adaptive_threshold else '') + '.txt'
    with open(file_name, "w") as file:
        file.write(f"Results : Dataset - {args.dataset}, Learning Rate - {args.lr}, Number of Workers - {args.nworker}, LocalIter - {args.localiter}\n")
        file.close()
    
    with open(checkpoint_file_name, "w") as file:
        file.write(f"Variance Checkpoints (max, s) : Dataset - {args.dataset}, Learning Rate - {args.lr}, Number of Workers - {args.nworker}, LocalIter - {args.localiter}\n")
        file.close()

    with open(outlier_file_name, "w") as file:
        file.write(f"Outlier removal results: Dataset - {args.dataset}, Learning Rate - {args.lr}, Number of Workers - {args.nworker}, LocalIter - {args.localiter}\n")
        file.close()

    for round_idx in range(args.round):
        print("Round: ", round_idx)
        
        with open(checkpoint_file_name, "a+") as file:
            file.write(f"Round: {round_idx}\n")
            file.close()

        with open(outlier_file_name, "a+") as file:
            file.write(f"Round: {round_idx}\n")
            file.close()

        choices = np.arange(args.nworker)
        if args.attack == 'modelpoisoning':
            dynamic_mal_index = np.random.choice(choices, args.malnum, replace=False)

        # copy network parameters
        params_copy = []
        for p in list(network.parameters()):
            params_copy.append(p.clone())

        for c in tqdm(choices):
            if args.attack == 'modelpoisoning' and c in dynamic_mal_index:
                for idx, p in enumerate(local_grads[c]):
                    local_grads[c][idx] = np.zeros(p.shape)

                for _ in range(0, args.localiter):
                    params_temp = []
                    for p in list(network.parameters()):
                        params_temp.append(p.clone())
                    delta_mal = mal_single(mal_train_loaders, train_loaders[c], network, criterion, optimizer, params_temp, device, [], round_idx, dist=True, mal_boost=2.0, path=args.agg)
                    for idx, p in enumerate(local_grads[c]):
                        local_grads[c][idx] = p + delta_mal[idx]
            
            elif c in mal_index and args.attack == 'backdoor':

                for idx, p in enumerate(local_grads[c]):
                    local_grads[c][idx] = np.zeros(p.shape)

                for _ in range(0, args.localiter):
                    for idx, (feature, target) in enumerate(train_loaders[c], 0):
                        attack_feature = (TF.erase(feature, 0, 0, 5, 5, 0).to(device))
                        attack_target = torch.zeros(args.batchsize, dtype=torch.long).to(device)
                        optimizer.zero_grad()
                        output = network(attack_feature)
                        loss = criterion(output, attack_target)
                        loss.backward()
                        optimizer.step()

                for idx, p in enumerate(network.parameters()):
                    local_grads[c][idx] = params_copy[idx].data.cpu().numpy() - p.data.cpu().numpy()

            else:
                for _ in range(0, args.localiter):
                    for idx, (feature, target) in enumerate(train_loaders[c], 0):
                        feature = feature.to(device)
                        target = target.type(torch.long).to(device)
                        optimizer.zero_grad()
                        output = network(feature)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()

                if args.agg == 'iclr2022_bucketing' or args.agg == 'icml2021_history':
                    for idx, p in enumerate(network.parameters()):
                        local_grads[c][idx] = (1 - args.beta) * (params_copy[idx].data.cpu().numpy() - p.data.cpu().numpy()) + args.beta * local_grads[c][idx]
                else:
                    for idx, p in enumerate(network.parameters()):
                        local_grads[c][idx] = params_copy[idx].data.cpu().numpy() - p.data.cpu().numpy()

            # manually restore the parameters of the global network
            with torch.no_grad():
                for idx, p in enumerate(list(network.parameters())):
                    p.copy_(params_copy[idx])

        grads = [[] for _ in range(len(local_grads[0]))]
            
        for i in range(len(local_grads[0])):
            for j in range(len(local_grads)):
                grads[i].append(local_grads[j][i].reshape(-1))
                
            grads[i] = np.vstack(grads[i])
        
        itv = 1000
        
        if args.adaptive_threshold:
            thresholds = []
            if args.agg in ['ex_noregret', 'filterl2'] or args.attack in ['variance_diff', 'single_direction', 'partial_single_direction']:
                for i in range(len(grads)):
                    feature_size = grads[i].shape[1]
                    cnt = int(feature_size // itv)
                    idx = 0
                    sizes = []
                    threshold_layer = []
                    for _ in range(cnt):
                        sizes.append(itv)
                    if feature_size % itv:
                        sizes.append(feature_size - cnt * itv)    
                    
                    for j in range(len(sizes)):
                        partitioned_grads = grads[i][:, idx:idx+itv]
                        cov = np.cov(partitioned_grads, rowvar=False, bias=True)
                        eigenvalues, _ = np.linalg.eigh(cov)
                                                
                        max_variance = np.max(np.abs(eigenvalues))
                        threshold_layer.append(max_variance)

                        idx = idx + itv

                    thresholds.append(threshold_layer)

        if args.attack == 'modelpoisoning':
            average_grad = []

            for p in list(network.parameters()):
                average_grad.append(np.zeros(p.data.shape))

            for c in choices:
                if c not in mal_index:
                    for idx, p in enumerate(average_grad):
                        average_grad[idx] = p + local_grads[c][idx] / args.perround

            np.save('./checkpoints/' + args.agg + 'ben_delta_t%s.npy' % round_idx, average_grad)

        # TODO: choices are not passed in as arguments, will the following two attacks deal with sub-sampled FL correctly?
        elif args.attack == 'trimmedmean':
            print('attack trimmedmean')
            local_grads = attack_trimmedmean(network, local_grads, mal_index, b=1.5)

        elif args.attack == 'krum':
            print('attack krum')
            for idx, _ in enumerate(local_grads[0]):
                local_grads = attack_krum(network, local_grads, mal_index, idx)

        elif args.attack == 'xie':
            print('attack Xie et al.')
            local_grads = attack_xie(local_grads, 1, choices, mal_index)


        if args.attack in ['variance_diff', 'single_direction', 'partial_single_direction']:
            for i in range(len(grads)):
                feature_size = grads[i].shape[1]
                cnt = int(feature_size // itv)
                idx = 0
                sizes = []
                for _ in range(cnt):
                    sizes.append(itv)
                if feature_size % itv:
                    sizes.append(feature_size - cnt * itv)

                for j in range(len(sizes)):
                    if args.adaptive_threshold:
                        threshold = thresholds[i][j]
                    else:
                        threshold = args.sigma

                    if args.attack == "variance_diff":
                        grads[i][:, idx:idx+itv] = corrupt_grads(grads[i][:, idx:idx+itv], mal_index, benign_index, threshold)
                    elif args.attack == "single_direction":
                        grads[i][:, idx:idx+itv] = attack_single_direction(grads[i][:, idx:idx+itv], mal_index, benign_index, threshold, args.attack, args.dataset, args.lr, device, checkpoint_file_name)
                    elif args.attack == "partial_single_direction":
                        grads[i][:, idx:idx+itv] = partial_attack_single_direction(grads[i][:, idx:idx+itv], mal_index, benign_index, threshold, args.attack, args.dataset, args.lr, device, checkpoint_file_name)

                    idx = idx + itv
            reshaped_grads = [[] for _ in range(len(grads[0]))]

            for i in range(len(grads)):
                for j in range(len(grads[i])):
                    reshaped_grads[j].append(grads[i][j].reshape(local_grads[j][i].shape))
            
            local_grads = reshaped_grads

        # aggregation
        average_grad = []

        for p in list(network.parameters()):
            average_grad.append(np.zeros(p.data.shape))

        if args.agg == 'average':
            print('agg: average')
            s = time.time()
            for idx, p in enumerate(average_grad):
                avg_local = []
                for c in choices:
                    avg_local.append(local_grads[c][idx])
                avg_local = np.array(avg_local)
                average_grad[idx] = np.average(avg_local, axis=0)
                
            print('average running time: ', time.time()-s)
        elif args.agg == 'krum':
            print('agg: krum')
            s = time.time()
            for idx, p in enumerate(average_grad):
                krum_local = []
                for c in choices:
                    krum_local.append(local_grads[c][idx])
                average_grad[idx], _ = krum(krum_local, f=args.malnum)
            # print('krum running time: ', time.time()-s)
        elif args.agg == 'filterl2':
            print('agg: filterl2')
            s = time.time()
            for idx, _ in enumerate(average_grad):
                filterl2_local = []
                for c in choices:
                    filterl2_local.append(local_grads[c][idx])

                if args.adaptive_threshold:
                    current_thresholds = thresholds[idx]
                else:
                    current_thresholds = None
        
                average_grad[idx] = filterL2(filterl2_local, eps=args.malnum*1./args.nworker, sigma=args.sigma, thresholds=current_thresholds, device=device, file_name=outlier_file_name)
            # print('filterl2 running time: ', time.time()-s)
        elif args.agg == 'mom_filterl2':
            print('agg: mom_filterl2')
            s = time.time()
            for idx, _ in enumerate(average_grad):
                filterl2_local = []
                for c in choices:
                    filterl2_local.append(local_grads[c][idx])
                average_grad[idx] = mom_filterL2(filterl2_local, eps=args.malnum*1./args.nworker, sigma=args.sigma, delta=np.exp(-50+args.malnum))
            # print('mom_filterl2 running time: ', time.time()-s)
        elif args.agg == 'median':
            print('agg: median')
            s = time.time()
            for idx, _ in enumerate(average_grad):
                median_local = []
                for c in choices:
                    median_local.append(local_grads[c][idx])
                average_grad[idx] = median(median_local)
            # print('median running time: ', time.time()-s)
        elif args.agg == 'trimmedmean':
            print('agg: trimmedmean')
            s = time.time()
            for idx, _ in enumerate(average_grad):
                trimmedmean_local = []
                for c in choices:
                    trimmedmean_local.append(local_grads[c][idx])
                average_grad[idx] = trimmed_mean(trimmedmean_local)
            # print('trimmedmean running time: ', time.time()-s)
        elif args.agg == 'bulyankrum':
            print('agg: bulyankrum')
            s = time.time()
            for idx, _ in enumerate(average_grad):
                bulyan_local = []
                for c in choices:
                    bulyan_local.append(local_grads[c][idx])
                average_grad[idx] = bulyan(bulyan_local, args.malnum, aggsubfunc='krum')
            # print('bulyankrum running time: ', time.time()-s)
        elif args.agg == 'bulyanmedian':
            print('agg: bulyanmedian')
            s = time.time()
            for idx, _ in enumerate(average_grad):
                bulyan_local = []
                for c in choices:
                    bulyan_local.append(local_grads[c][idx])
                average_grad[idx] = bulyan(bulyan_local, args.malnum, aggsubfunc='median')
            # print('bulyanmedian running time: ', time.time()-s)
        elif args.agg == 'bulyantrimmedmean':
            print('agg: bulyantrimmedmean')
            s = time.time()
            for idx, _ in enumerate(average_grad):
                bulyan_local = []
                for c in choices:
                    bulyan_local.append(local_grads[c][idx])
                average_grad[idx] = bulyan(bulyan_local, args.malnum, aggsubfunc='trimmedmean')
            # print('bulyantrimmedmean running time: ', time.time()-s)
        elif args.agg == 'ex_noregret':
            print('agg: explicit non-regret')
            s = time.time()
            for idx, _ in enumerate(average_grad):
                ex_noregret_local = []
                for c in choices:
                    ex_noregret_local.append(local_grads[c][idx])
                average_grad[idx] = ex_noregret(ex_noregret_local, eps=args.malnum*1./args.nworker, sigma=args.sigma)
            # print('ex_noregret running time: ', time.time()-s)
        elif args.agg == 'mom_ex_noregret':
            print('agg: mom explicit non-regret')
            s = time.time()
            for idx, _ in enumerate(average_grad):
                ex_noregret_local = []
                for c in choices:
                    ex_noregret_local.append(local_grads[c][idx])
                average_grad[idx] = mom_ex_noregret(ex_noregret_local, eps=args.malnum*1./args.nworker, sigma=args.sigma, delta=np.exp(-50+args.malnum))
            # print('mom_ex_noregret running time: ', time.time()-s)
        elif args.agg == 'iclr2022_bucketing':
            print('agg: BYZANTINE-ROBUST LEARNING ON HETEROGENEOUS DATASETS VIA BUCKETING ICLR 2022')
            s = time.time()
            if prev_average_grad is None:
                prev_average_grad = []
                for p in list(network.parameters()):
                    prev_average_grad.append(np.zeros(p.data.shape))
                    shuffled_choices = np.random.shuffle(choices)
            bucket_average_grads = []
            for bidx in range(args.buckets):
                bucket_average_grads.append([])
                for idx, _ in enumerate(average_grad):
                    avg_local = []
                    for c in choices[bidx: bidx+args.perround//args.buckets]:
                        avg_local.append(local_grads[c][idx])
                    avg_local = np.array(avg_local)
                    bucket_average_grads[bidx].append(np.average(avg_local, axis=0))
            for bidx in range(args.buckets):
                norm = 0.
                for idx, _ in enumerate(average_grad):
                    norm += np.linalg.norm(bucket_average_grads[bidx][idx] - prev_average_grad[idx])**2
                norm = np.sqrt(norm)
                for idx, _ in enumerate(average_grad):
                    bucket_average_grads[bidx][idx] = (bucket_average_grads[bidx][idx] - prev_average_grad[idx]) * min(1, args.tau/norm)
            for idx, _ in enumerate(average_grad):
                avg_local = []
                for bidx in range(args.buckets):
                    avg_local.append(bucket_average_grads[bidx][idx])
                avg_local = np.array(avg_local)
                average_grad[idx] = np.average(avg_local, axis=0)
            prev_average_grad = copy.deepcopy(average_grad)
            # print('iclr2022_bucketing running time: ', time.time()-s)
        elif args.agg == 'icml2021_history':
            print('agg: Learning from History for Byzantine Robust Optimization ICML 2021')
            s = time.time()
            if prev_average_grad is None:
                prev_average_grad = []
                for p in list(network.parameters()):
                    prev_average_grad.append(np.zeros(p.data.shape))
            for c in choices:
                norm = 0.
                for idx, _ in enumerate(average_grad):
                    norm += np.linalg.norm(local_grads[c][idx] - prev_average_grad[idx])**2
                norm = np.sqrt(norm)
                for idx, _ in enumerate(average_grad):
                    local_grads[c][idx] = (local_grads[c][idx] - prev_average_grad[idx]) * min(1, args.tau/norm)
            for idx, _ in enumerate(average_grad):
                avg_local = []
                for c in choices:
                    avg_local.append(local_grads[c][idx])
                avg_local = np.array(avg_local)
                average_grad[idx] = np.average(avg_local, axis=0)
            prev_average_grad = copy.deepcopy(average_grad)
            # print('icml2021_history running time: ', time.time()-s)
        elif args.agg == 'clustering':
            print('agg: Secure Byzantine-Robust Distributed Learning via Clustering')
            s = time.time()
            for idx, _ in enumerate(average_grad):
                avg_local = []
                for c in choices:
                    avg_local.append(local_grads[c][idx])
                average_grad[idx] = mom_krum(avg_local, f=args.malnum)
            # print('clustering running time: ', time.time()-s)


        params = list(network.parameters())
        with torch.no_grad():
            for idx in range(len(params)):
                grad = torch.from_numpy(average_grad[idx]).to(device)
                params[idx].data.sub_(grad)


        if (round_idx + 1) % args.checkpoint == 0:
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for feature, target in test_loader:
                    feature = feature.to(device)
                    target = target.type(torch.long).to(device)
                    output = network(feature)
                    test_loss += F.cross_entropy(output, target, reduction='sum').item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            accuracy = 100. * correct / len(test_loader.dataset)

            if args.attack == 'modelpoisoning':

                mal_test_loss = 0
                correct = 0
                with torch.no_grad():
                    for idx, (feature, mal_data, true_label, target) in enumerate(mal_train_loaders, 0):
                        feature = feature.to(device)
                        target = target.type(torch.long).to(device)
                        output = network(feature)
                        mal_test_loss += F.nll_loss(output, target, reduction='sum').item()
                        pred = output.data.max(1, keepdim=True)[1]
                        correct += pred.eq(target.data.view_as(pred)).sum()
                mal_test_loss /= len(mal_train_loaders.dataset)
                # txt_file.write('%d, \t%f, \t%f, \t%f, \t%f\n'%(round_idx+1, test_loss, accuracy, mal_test_loss, 100. * correct / len(mal_train_loaders.dataset)))
                print('\nMalicious set: Accuracy: {:.4f}, Attack Success Rate: {}/{} ({:.0f}%)\n'.format(accuracy, correct, len(mal_train_loaders.dataset), 100. * correct / len(mal_train_loaders.dataset)))
            

            elif args.attack == 'backdoor':

                mal_test_loss = 0
                correct = 0
                with torch.no_grad():
                    for feature, target in test_loader:
                        feature = (TF.erase(feature, 0, 0, 5, 5, 0).to(device))
                        target = torch.zeros(1, dtype=torch.long).to(device)
                        output = network(feature)
                        mal_test_loss += F.nll_loss(output, target, size_average=False).item()
                        pred = output.data.max(1, keepdim=True)[1]
                        correct += pred.eq(target.data.view_as(pred)).sum()
                test_loss /= len(test_loader.dataset)
                # txt_file.write('%d, \t%f, \t%f, \t%f, \t%f\n'%(round_idx+1, test_loss, accuracy, mal_test_loss, 100. * correct / len(test_loader.dataset)))
                print('\nMalicious set: Accuracy: {:.4f}, Attack Success Rate: {}/{} ({:.0f}%)\n'.format(accuracy, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


            else:
                # txt_file.write('%d, \t%f, \t%f\n'%(round_idx+1, test_loss, accuracy))
                with open(file_name, "a+") as file:
                    file.write('%d, \t%f, \t%f\n'%(round_idx+1, test_loss, accuracy))
                    file.close()

                print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), accuracy))
