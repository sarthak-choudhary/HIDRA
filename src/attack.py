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
    This file contains the attack functions on Federated Learning.
'''

import copy
import numpy as np
import random
from robust_estimator import krum
from scipy.spatial import distance
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF

# Trimmed Mean Attack--Main function for TMA.
def attack_trimmedmean(network, local_grads, mal_index, b=2):
    benign_max = []
    benign_min = []
    average_sign = []
    mal_param = []
    local_param = copy.deepcopy(local_grads)
    for i in sorted(mal_index, reverse=True):
        del local_param[i]
    for p in list(network.parameters()):
        benign_max.append(np.zeros(p.data.shape))
        benign_min.append(np.zeros(p.data.shape))
        average_sign.append(np.zeros(p.data.shape))
        mal_param.append(np.zeros(p.data.shape))
    for idx, p in enumerate(average_sign):
        for c in range(len(local_param)):
            average_sign[idx] += local_param[c][idx]
        average_sign[idx] = np.sign(average_sign[idx])
    for idx, p in enumerate(network.parameters()):
        temp = []
        for c in range(len(local_param)):
            local_param[c][idx] = p.data.cpu().numpy() - local_param[c][idx]
            temp.append(local_param[c][idx])
        temp = np.array(temp)
        benign_max[idx] = np.amax(temp, axis=0)
        benign_min[idx] = np.amin(temp, axis=0)
    
    for idx, p in enumerate(average_sign):
        for aver_sign, b_max, b_min, mal_p in np.nditer([p, benign_max[idx], benign_min[idx], mal_param[idx]], op_flags=['readwrite']):
            if aver_sign < 0:
                if b_min > 0:
                    mal_p[...] = random.uniform(b_min/b, b_min)
                else:
                    mal_p[...] = random.uniform(b_min*b, b_min)
            else:
                if b_max > 0:
                    mal_p[...] = random.uniform(b_max, b_max*b)
                else:
                    mal_p[...] = random.uniform(b_max, b_max/b)
    for c in mal_index:
        for idx, p in enumerate(network.parameters()):
            local_grads[c][idx] = -mal_param[idx] + p.data.cpu().numpy()
    return local_grads


# Krum Attack--Main function for KA.
def attack_krum(network, local_grads, mal_index, param_index, lower_bound=1e-8, upper_bound=1e-3):

    local_param = copy.deepcopy(local_grads)
    for i in sorted(mal_index, reverse=True):
        del local_param[i]
    m = len(local_grads)
    c = len(mal_index)
    d = local_grads[0][param_index].size

    average_sign = np.zeros(list(network.parameters())[param_index].data.shape)
    benign_max = np.zeros(list(network.parameters())[param_index].data.shape)

    for c in range(len(local_param)):
        average_sign += local_param[c][param_index]
    average_sign  = np.sign(average_sign)
    min_dis = np.inf
    max_dis = -np.inf
    for i in range(m):
        if i in mal_index:
            continue
        else:
            temp_min_dis = 0
            temp_max_dis = 0
            for j in range(m):
                if j in mal_index or j == i:
                    continue
                else:
                    temp_min_dis += distance.euclidean(local_grads[i][param_index].flatten(), local_grads[j][param_index].flatten())
        temp_max_dis += distance.euclidean(local_grads[i][param_index].flatten(), benign_max.flatten())

        if temp_min_dis < min_dis:
            min_dis = temp_min_dis
        if temp_max_dis > max_dis:
            max_dis = temp_max_dis
    
    upper_bound = 1.0
    lambda1 = upper_bound

    if upper_bound < lower_bound:
        print('Wrong lower bound!')

    while True:
        krum_local = []
        for kk in range(len(local_grads)):
            krum_local.append(local_grads[kk][param_index])
        for kk in mal_index:
            krum_local[kk] = -lambda1 * average_sign
        _, choose_index = krum(krum_local, f=1)
        if choose_index in mal_index:
            print('found a lambda')
            break
        elif lambda1 < lower_bound:
            print(choose_index, 'Failed to find a proper lambda!')
            break
        else:
            lambda1 /= 2.0
    
    for kk in mal_index:
        local_grads[kk][param_index] = -lambda1 * average_sign

    return local_grads

def bulyan_attack_krum(network, local_grads, mal_index, param_index, lower_bound=1e-8, upper_bound=1e-3, target_layer=0, target_idx=0):

    benign_max = []
    attack_vec = []

    local_param = copy.deepcopy(local_grads)
    for i in sorted(mal_index, reverse=True):
        del local_param[i]
    m = len(local_grads)
    c = len(mal_index)
    d = local_grads[0][param_index].size
    for p in list(network.parameters()):
        benign_max.append(np.zeros(p.data.shape))
        attack_vec.append(np.zeros(p.data.shape))

    for idx, p in enumerate(attack_vec):
        for c in range(len(local_param)):
            if c == target_layer and idx == target_idx:
                attack_vec[idx] += 1
 
    upper_bound = 1.0
    lambda1 = upper_bound

    if upper_bound < lower_bound:
        print('Wrong lower bound!')

    while True:
        krum_local = []
        for kk in range(len(local_grads)):
            krum_local.append(local_grads[kk][param_index])
        for kk in mal_index:
            krum_local[kk] = -lambda1 * attack_vec[param_index]
        _, choose_index = krum(krum_local, f=1)
        if choose_index in mal_index:
            break
        elif lambda1 < lower_bound:
            print(choose_index, 'Failed to find a proper lambda!')
            break
        else:
            lambda1 /= 2.0
    
    for kk in mal_index:
        local_grads[kk][param_index] = -lambda1 * attack_vec[param_index]
    
    return local_grads

# Inner-Product Manipulation Attack--Main function for IMA.
def attack_xie(local_grads, weight, choices, mal_index):
    attack_vec = []
    for i, pp in enumerate(local_grads[0]):
        tmp = np.zeros_like(pp)
        for ji, j in enumerate(choices):
            if j not in mal_index:
                tmp += local_grads[j][i]
        attack_vec.append((-weight) * tmp / len(choices))
    for i in mal_index:
        local_grads[i] = attack_vec
    return local_grads

# Partial Knowledge HIDRA--Main function for Partial Knowledge HIDRA
def partial_attack_single_direction(grads, corrupted_indices, benign_indices, threshold):
    num_corruptions = len(corrupted_indices) * 1.0
    num_benign = len(benign_indices) * 1.0

    eps = num_corruptions / (num_corruptions + num_benign)

    benign_mean = np.mean(grads[corrupted_indices], axis=0)

    if np.linalg.norm(benign_mean) == 0:
        return grads

    s = benign_mean / np.linalg.norm(benign_mean)

    variance_diff = (np.sqrt(20) - 1) * threshold

    corruption  = np.sqrt(1/(eps*eps + (1-eps)*(1-eps)*eps)) * np.sqrt(variance_diff)
    grads[corrupted_indices] = benign_mean - s*corruption
    return grads

# Full Knowledge HIDRA--Main function for Full Knowledge HIDRA
def attack_single_direction(grads, corrupted_indices, benign_indices, threshold):
    num_corruptions = len(corrupted_indices) * 1.0
    num_benign = len(benign_indices) * 1.0
    
    eps = num_corruptions / (num_corruptions + num_benign)

    benign_mean = np.mean(grads, axis=0)

    if np.linalg.norm(benign_mean) == 0:
        return grads
    
    s = benign_mean / np.linalg.norm(benign_mean)

    variance_diff = (np.sqrt(20) - 1) * threshold

    corruption  = np.sqrt(1/(eps*eps + (1-eps)*(1-eps)*eps)) * np.sqrt(variance_diff)
    grads[corrupted_indices] = benign_mean - s*corruption
    return grads