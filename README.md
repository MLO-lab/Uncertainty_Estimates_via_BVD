# Uncertainty Estimates of Predictions via a General Bias-Variance Decomposition

The official source code to [Uncertainty Estimates of Predictions via a General Bias-Variance Decomposition (AISTATS'23)](https://arxiv.org/abs/2210.12256).

## Code for Bregman divergence and Bregman Information generated by LogSumExp

For quality of life, the following Pytorch implementation should easily work via Copy-Pasting.
It differs slightly from the experiment code, since there were unexpectedly positive mathematical results after the experiments had finished.

```python
import torch

def BI_LSE(zs, axis=0, class_axis=-1):
    '''
    Bregman Information of random variable Z generated by G = LSE
    BI_G [ Z ] = E[ G( Z ) ] - G( E[ Z ] )
    We estimate with dataset zs = [Z_1, ..., Z_n] via
    1/n sum G( Z_i ) - G( 1/n sum Z_i )
    '''
    E_of_LSE = zs.logsumexp(axis=class_axis).mean(axis)
    LSE_of_E = zs.mean(axis).unsqueeze(axis).logsumexp(axis=class_axis).squeeze(axis)
    return E_of_LSE - LSE_of_E

def inner_product(a, b):
    ''' Batch wise inner product of last axis in a and b'''
    n_size, n_classes = a.shape
    return torch.bmm(a.view(n_size, 1, n_classes), b.view(n_size, n_classes, 1)).squeeze(-1).squeeze(-1)

def d_LSE(a, b):
    '''
    Bregman divergence generated by G = LSE
    d_G (a, b) = G(b) - G(a) - < grad G(a), b - a >
    We assume the classes are in the last axis.
    a: n x p
    b: n x p
    output: n x n
    '''
    G_of_a = a.logsumexp(axis=-1)
    G_of_b = b.logsumexp(axis=-1)
    grad_G_of_a = a.softmax(axis=-1)
    return G_of_b - G_of_a - inner_product(grad_G_of_a, b - a)
```

## Experiments

All experiments are run and plotted in Jupyter notebooks.
Installing the full environment might only be necessary for the CIFAR10 and ImageNet experiments.

### Environment Setup

The following allows to create and to run a python environment with all required dependencies using [miniconda](https://docs.conda.io/en/latest/miniconda.html): 

```(bash)
conda env create -f environment.yml
conda activate UQ
```

### Iris Confidence Regions (Figure 4)

The experiments for the confidence regions of the Iris classifier (Figure 4) can be found in `CR_iris.ipynb`.
They are done via Pytorch and are computationally light-weight (should run on any laptop).

### Classifiers on Toy Simulations (Figure 5 & 6)

We train and evaluate SK-Learn classifiers on toy simulations in `toy_simulations.ipynb`.
They are feasible to run locally on a laptop (they should finish in less than an hour).

### ResNet on CIFAR10 and CIFAR10-C (Figure 1)

These experiments can be found in `CIFAR10_ResNet.ipynb`.
They are expensive to evaluate and require a basic GPU.
The weight initialization ensembles are locally trained.
The data is supposed to be stored in `../data/`.

### ResNet on ImageNet and ImageNet-C (Figure 7)

These experiments can be found in `ImageNet_ResNet.ipynb`.
They are very expensive to evaluate and require an advanced GPU.
The weight initialization ensembles are taken from https://github.com/SamsungLabs/pytorch-ensembles.
For this, download the folder `deepens_imagenet` from [here](https://disk.yandex.ru/d/qwwESfJkkO48Bw?w=1) and extract it into a folder `../saved_models/`.
This can be either done manually or by
```
pip3 install wldhx.yadisk-direct

% if folder does not exist yet
mkdir ../saved_models

% ImageNet
curl -L $(yadisk-direct https://yadi.sk/d/rdk6ylF5mK8ptw?w=1) -o ../saved_models/deepens_imagenet.zip
unzip deepens_imagenet.zip 
```
The data is supposed to be stored in `../data/`.

## Attribution

- [PyTorch](https://github.com/pytorch/pytorch)

## Citation

```
@misc{gruber2023uncertainty,
      title={Uncertainty Estimates of Predictions via a General Bias-Variance Decomposition}, 
      author={Sebastian G. Gruber and Florian Buettner},
      year={2023},
      eprint={2210.12256},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
