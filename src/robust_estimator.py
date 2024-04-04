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
    Robust estimator for Federated Learning.
'''
import numpy as np
import torch
from torch.distributions.kl import kl_divergence

MAX_ITER = 100
ITV = 1000

def ex_noregret_(samples, eps=1./12, sigma=1, expansion=20, dis_threshold=0.7):
    """
    samples: data samples in tensor
    sigma: operator norm of covariance matrix assumption
    """
    dis_list = []

    for i in range(size):
        for j in range(i+1, size):
            distance = torch.norm(samples[i] - samples[j])
            dis_list.append(distance.item())

    dis_list = torch.tensor(dis_list).to(samples.device)
    if torch.max(dis_list) != 0:
        step_size = 0.5/(torch.max(dis_list)**2)
    else:
        step_size = torch.tensor(1)
    step_size = step_size.to(samples.device)

    size = samples.shape[0]
    feature_size = samples.shape[1]
    samples_ = samples.reshape(size, 1, feature_size)

    # c = np.ones(size)
    c = torch.ones(size).to(samples.device)
    for i in range(int(2 * eps * size)):
        avg = torch.sum(samples * c[:, None], dim=0) / torch.sum(c)
        cov = torch.cov(samples.T, correction=0)

        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov, eigenvectors=True)
        except:
            eigenvalues, eigenvectors = torch.linalg.eig(cov)
            eigenvalues = eigenvalues.real
            eigenvectors = eigenvectors.real

        eig_val = torch.abs(eigenvalues[0])
        eig_vec = eigenvectors[:, 0]

        if eig_val.item() * eig_val.item() <= expansion * sigma * sigma:
            return avg.cpu().numpy()

        tau = torch.tensor([torch.dot(sample - avg, eig_vec).pow(2) for sample in samples]).to(samples.device)
        c = c * (1 - step_size * tau)

        # The projection step
        ordered_c_index = torch.argsort(c, descending=True)
        min_KL = None
        projected_c = None
        for i in range(len(c)):
            c_ = c.clone()
            for j in range(i+1):   
                c_[ordered_c_index[j]] = 1./(1-eps)/len(c)

            clip_norm = 1 - torch.sum(c_[ordered_c_index[:i+1]])
            norm = torch.sum(c_[ordered_c_index[i+1:]])
            if clip_norm <= 0:
                break
            scale = clip_norm / norm
            for j in range(i+1, len(c)):
                c_[ordered_c_index[j]] = c_[ordered_c_index[j]] * scale
            if c_[ordered_c_index[i+1]] > 1./(1-eps)/len(c):
                continue
            KL = torch.sum(kl_divergence(torch.distributions.Categorical(probs=c), torch.distributions.Categorical(probs=c_)))
            # KL = np.sum(rel_entr(c, c_))
            if min_KL is None or KL < min_KL:
                min_KL = KL
                projected_c = c_

        c = projected_c

    avg = torch.sum(samples * c[:, None], dim=0) / torch.sum(c)
    return avg.cpu().numpy()

def ex_noregret(samples, eps=1./12, sigma=1, expansion=20, itv=ITV, device="cpu"):
    """
    samples: data samples in numpy array
    sigma: operator norm of covariance matrix assumption
    """
    samples = np.array(samples)
    feature_shape = samples[0].shape
    samples_flatten = []
    for i in range(samples.shape[0]):
        samples_flatten.append(samples[i].flatten())
    samples_flatten = np.array(samples_flatten)
    feature_size = samples_flatten.shape[1]
    if itv is None:
        itv = int(np.floor(np.sqrt(feature_size)))
    cnt = int(feature_size // itv)
    sizes = []
    for i in range(cnt):
        sizes.append(itv)
    if feature_size % itv:
        sizes.append(feature_size - cnt * itv)

    idx = 0
    res = []
    cnt = 0
    for size in sizes:
        cnt += 1
        partitioned_samples = torch.tensor(samples_flatten[:,idx:idx+size]).to(device)
        res.append(ex_noregret_(partitioned_samples, eps, sigma, expansion))
        idx += size

    return np.concatenate(res, axis=0).reshape(feature_shape)

def filterL2_(samples, eps=0.2, sigma=1, expansion=20):
    """
    samples: data samples in numpy array
    sigma: operator norm of covariance matrix assumption
    """
    size = samples.shape[0]
    feature_size = samples.shape[1]

    samples_ = samples.reshape(size, 1, feature_size)

    c = np.ones(size)
    points_removed = []
    for i in range(2 * int(eps * size)):
        # print(i)
        avg = np.average(samples, axis=0, weights=c)
        cov = np.cov(samples, rowvar=False, bias=True)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        max_eigenvalue_index = np.argmax(np.abs(eigenvalues))
        eig_val = eigenvalues[max_eigenvalue_index]
        eig_vec = eigenvectors[:, max_eigenvalue_index]

        if eig_val * eig_val <= expansion * sigma * sigma:
            return avg
        
        tau = np.array([np.inner(sample-avg, eig_vec)**2 for sample in samples])
        tau_max_idx = np.argmax(tau)
        tau_max = tau[tau_max_idx]
        c = c * (1 - tau/tau_max)

        samples = np.concatenate((samples[:tau_max_idx], samples[tau_max_idx+1:]))
        points_removed.append(tau_max_idx)
        samples_ = samples.reshape(-1, 1, feature_size)
        c = np.concatenate((c[:tau_max_idx], c[tau_max_idx+1:]))
        c = c / np.linalg.norm(c, ord=1)
            
    avg = np.average(samples, axis=0, weights=c)
    return avg

 
def filterL2(samples, eps=0.2, sigma=1, expansion=20, itv=ITV, thresholds=None, device="cpu"):
    """
    samples: data samples in numpy array
    sigma: operator norm of covariance matrix assumption
    """
    samples = np.array(samples)
    feature_shape = samples[0].shape
    samples_flatten = []
    for i in range(samples.shape[0]):
        samples_flatten.append(samples[i].flatten())
    samples_flatten = np.array(samples_flatten)
    # print(samples_flatten.shape)
    feature_size = samples_flatten.shape[1]
    if itv is None:
        itv = int(np.floor(np.sqrt(feature_size)))
    cnt = int(feature_size // itv)
    sizes = []
    for i in range(cnt):
        sizes.append(itv)
    if feature_size % itv:
        sizes.append(feature_size - cnt * itv)

    idx = 0
    res = []
    
    for i in range(len(sizes)):
        partitioned_samples = samples_flatten[:,idx:idx+sizes[i]]
        
        if thresholds:
            threshold = thresholds[i]
        else:
            threshold = sigma

        res.append(filterL2_(partitioned_samples, eps, threshold, expansion))
        idx += sizes[i]

    return np.concatenate(res, axis=0).reshape(feature_shape)

def median(samples):
    return np.median(samples, axis=0)

def trimmed_mean(samples, beta=0.1):
    samples = np.array(samples)
    average_grad = np.zeros(samples[0].shape)
    size = samples.shape[0]
    beyond_choose = int(size * beta)
    samples = np.sort(samples, axis=0)
    samples = samples[beyond_choose:size-beyond_choose]
    average_grad = np.average(samples, axis=0)

    return average_grad

def krum_(samples, f):
    size = len(samples)
    size_ = size - f - 2
    metric = []
    for idx in range(size):
        sample = samples[idx]
        samples_ = samples.copy()
        del samples_[idx]
        dis = np.array([np.linalg.norm(sample-sample_) for sample_ in samples_])
        metric.append(np.sum(dis[np.argsort(dis)[:size_]]))
    return metric

def krum(samples, f):
    metric = krum_(samples, f)
    index = np.argmin(metric)
    return samples[index], index

def bulyan_median(arr):
    arr_len = len(arr)
    distances = np.zeros([arr_len, arr_len])
    for i in range(arr_len):
        for j in range(arr_len):
            if i < j:
                distances[i, j] = abs(arr[i] - arr[j])
            elif i > j:
                distances[i, j] = distances[j, i]
    total_dis = np.sum(distances, axis=-1)
    median_index = np.argmin(total_dis)
    return median_index, distances[median_index]

def bulyan_one_coordinate(arr, beta):
    _, distances = bulyan_median(arr)
    median_beta_neighbors = arr[np.argsort(distances)[:beta]]
    return np.mean(median_beta_neighbors)

def bulyan(grads, f, aggsubfunc='trimmedmean'):
    samples = np.array(grads)
    feature_shape = grads[0].shape
    samples_flatten = []
    for i in range(samples.shape[0]):
        samples_flatten.append(samples[i].flatten())

    grads_num = len(samples_flatten)
    theta = grads_num - 2 * f
    # bulyan cannot do the work here when theta <= 0. Actually, it assumes n >= 4 * f + 3
    selected_grads = []
    # here, we use krum as sub algorithm
    if aggsubfunc == 'krum':
        for i in range(theta):
            krum_grads, _ = krum(samples_flatten, f)
            selected_grads.append(krum_grads)
            for j in range(len(samples_flatten)):
                if samples_flatten[j] is krum_grads:
                    del samples_flatten[j]
                    break
    elif aggsubfunc == 'median':
        for i in range(theta):
            median_grads = median(samples_flatten)
            selected_grads.append(median_grads)
            min_dis = np.inf
            min_index = None
            for j in range(len(samples_flatten)):
                temp_dis = np.linalg.norm(median_grads - samples_flatten[j])
                if temp_dis < min_dis:
                    min_dis = temp_dis
                    min_index = j
            assert min_index != None
            del samples_flatten[min_index]
    elif aggsubfunc == 'trimmedmean':
        for i in range(theta):
            trimmedmean_grads = trimmed_mean(samples_flatten)
            selected_grads.append(trimmedmean_grads)
            min_dis = np.inf
            min_index = None
            for j in range(len(samples_flatten)):
                temp_dis = np.linalg.norm(trimmedmean_grads - samples_flatten[j])
                if temp_dis < min_dis:
                    min_dis = temp_dis
                    min_index = j
            assert min_index != None
            del samples_flatten[min_index]

    beta = theta - 2 * f
    np_grads = np.array([g.flatten().tolist() for g in selected_grads])

    grads_dim = len(np_grads[0])
    selected_grads_by_cod = np.zeros([grads_dim, 1])  # shape of torch grads
    for i in range(grads_dim):
        selected_grads_by_cod[i, 0] = bulyan_one_coordinate(np_grads[:, i], beta)

    return selected_grads_by_cod.reshape(feature_shape)

def dnc_aggr(samples, niters=1, eps=0.2):
    n, d = samples.shape
    inliers = []
    outliers = []
    b = d

    for _ in range(niters):
        random_indices = np.random.permutation(d)[:b]
        random_indices.sort()
        sub_samples = samples[:, random_indices]
        sample_mean = np.mean(sub_samples, axis=0)
        centered_samples = sub_samples - sample_mean

        cov_matrix = np.cov(samples, rowvar=False, bias=True)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        max_eigenvalue_index = np.argmax(np.abs(eigenvalues))
        max_eigenvector = eigenvectors[:, max_eigenvalue_index]

        outlier_scores = []

        for i in range(n):
            outlier_score = np.dot(centered_samples[i], max_eigenvector)
            outlier_score = outlier_score * outlier_score

            outlier_scores.append(outlier_score)

        inliers.append(np.argsort(outlier_scores)[:int((1-eps)*n)].tolist())
        outliers.append(np.argsort(outlier_scores)[int((1-eps)*n):].tolist())
    
    set_inliers = [set(inlier) for inlier in inliers]

    inlier_indices = set.intersection(*set_inliers)
    inlier_indices = list(inlier_indices)

    inlier_samples = samples[inlier_indices, :]
    inlier_mean = np.mean(inlier_samples, axis=0)

    return inlier_mean, outliers