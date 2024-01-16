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

import numpy as np

def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")

def power_iteration(grads):
    rand_vector = np.random.randn(grads.shape[1])
    rand_vector = rand_vector / np.linalg.norm(rand_vector)
    
    num_iterations = 100
    current_eigenvector = rand_vector
    mean = np.mean(grads, axis=0)
    for itr in range(num_iterations):
        next_eigenvector = np.zeros(current_eigenvector.shape)
        for i in range(grads.shape[0]):
            next_eigenvector += np.dot(grads[i] - mean, current_eigenvector) * (grads[i] - mean)
        
        next_eigenvector = next_eigenvector/grads.shape[0]
        next_eigenvector = next_eigenvector/np.linalg.norm(next_eigenvector)
        current_eigenvector = next_eigenvector
        
    print(f"Start Variance: {compute_variance(grads, rand_vector)}   End Variance: {compute_variance(grads, current_eigenvector)}")
    return current_eigenvector

def compute_variance(grads, proj_vector):
    projections = [np.dot(grad, proj_vector) for grad in grads]
    variance = np.var(projections)
    return variance

class Record(object):
    def __init__(self):
        self.loss = 0
        self.count = 0

    def add(self, value):
        self.loss += value
        self.count += 1

    def mean(self):
        return self.loss / self.count
    