# Attacking Byzantine Robust Aggregation in High Dimensions

This repository serves as an evaluation of a novel untargeted model poisoning attack, HIDRA, from the manuscript titled "Attacking Byzantine Robust Aggregation in High Dimensions" by Sarthak Choudhary*, Aashish Kolluri*, and Prateek Saxena.

To achieve this, we have incorporated and adapted certain baseline implementations from [secure-robust-federated-learning ](https://github.com/wanglun1996/secure-robust-federated-learning) as a foundation for our evaluation. We extend our gratitude to the contributors of that repository for their valuable work

### Attacks 
In addition to assessing HIDRA, we have evaluated several other untargeted model poisoning attacks in the context of federated learning. This diverse set of attacks serves as a benchmark to establish the supremacy and effectiveness of HIDRA.
- [Krum](https://dl.acm.org/doi/abs/10.5555/3489212.3489304)
- [Trimmed Mean](https://dl.acm.org/doi/abs/10.5555/3489212.3489304)
- [Inner Product Manipulation](https://arxiv.org/abs/1903.03936)

### Byzantine-Robust Aggregators 
We employed various Byzantine-robust aggregators to rigorously test the effectiveness of different untargeted model poisoning attacks, including HIDRA. Our findings reveal that HIDRA emerges as uniquely effective against state-of-the-art Byzantine-robust aggregators. This underscores the significance of understanding and mitigating the specific threats posed by HIDRA in the context of federated learning.

- [Filtering](http://arxiv.org/abs/2205.11765)
- [No-regret](http://arxiv.org/abs/2205.11765)
- [Krum](https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html)
- [Median](https://proceedings.mlr.press/v80/yin18a)
- [Trimmed Mean](https://proceedings.mlr.press/v80/yin18a)

We suggest using Anaconda for managing the environment. 
## Setting up conda
First, create a conda virtual environment with Python 3.7.11 and activate the environment

```bash
conda create -n venv python=3.7.11
conda activate venv
```

Run the following command to install all the required python packages.

```bash
pip install -r requirements.txt
```


## Usage
Reproduce the evaluation results for HIDRA by running the following script.
```bash
./train.sh
```
To run a single Byzantine-robust [aggregator] against a single [attack] on a [dataset], run the following command with the right system arguments:  
```bash
python src/simulate.py --dataset [dataset] --attack [attack] --agg [aggregator]
```
Each run will store its evaluation result in `./results` directory. 
This is the full list of arguments for the aforementioned command. 
| Argument | Values | Use |
|----------|--------|-----|
|--dataset|`MNIST`, `FMNIST`, `CIFAR10`| dataset for evaluation|
|--agg|`average`, `filterl2`, `ex_noregret` ...| robust aggregator for evaluation|
|--attack|`single_direction`, `partial_single_direction` ...| attack for evaluation, use `single_direction` and `partial_single_direction` for evaluating HIDRA in full and partial knowledge respectively|
|--nworkers|`int`| number of clients for federated learning, default `100`|
|--malnum|`int`| number of malicious clients, default `20`|
|--perround|`int`| number of clients participating in each round of training, default `100`|
|--localiter|`int`| number of local iterations at each client to compute gradients, default `5`|
|--round|`int`| number of training rounds, defualt `100`|
|--lr|`float`| learning rate of the model, default `1e-3`|
|--sigma|`float`| variance threshold used in aggregators like `Filtering` and `No-Regret`, default `1e-5`|
|--batchsize|`int`| batchsize for training at each client|
|--adaptive_threshold| `-`| add to use adaptive variance threshold for `Filtering` and `No-Regret`|


## Acknowledgement

The code of evaluation of baselines largely reuse [secure-robust-federated-learning](https://github.com/wanglun1996/secure-robust-federated-learning).
