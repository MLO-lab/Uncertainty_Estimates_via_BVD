# Uncertainty Estimates of Predictions via a General Bias-Variance Decomposition

The official source code to [Uncertainty Estimates of Predictions via a General Bias-Variance Decomposition (AISTATS'23)](https://arxiv.org/abs/2210.12256).

## Code for Bregman divergence and Bregman Information generated by LogSumExp

We provide Pytorch and Numpy implementations, which should easily work via Copy-Pasting.
The exact implementation differs slightly from the experiments, since there were unexpectedly positive mathematical results after the experiments had finished.

### Pytorch

```
import torch

def BI_LSE(zs, axis=0, class_axis=-1):
    '''
    Bregman Information of random variable Z generated by G = LSE
    BI_G [ Z ] = E[ G( Z ) ] - G( E[ Z ] )
    We estimate via
    1/n sum G( Z_i ) - G( 1/n sum Z_i )
    '''
    E_of_LSE = zs.logsumexp(axis=class_axis).mean(axis).unsqueeze(axis)
    LSE_of_E = zs.mean(axis).unsqueeze(axis).logsumexp(axis=class_axis)
    return E_of_LSE - LSE_of_E

def inner_product(a, b):
    ''' batch wise inner product of last axis in a and b'''
    n_size, n_classes = a.shape
    return torch.bmm(a.view(n_size, 1, n_classes), b.view(n_size, n_classes, 1)).squeeze(-1).squeeze(-1)

def d_LSE(a, b):
    '''
    bregman divergence generated by G = LSE
    d_G (a, b) = G(a) - G(b) - < grad G(b), a - b >
    '''
    # assume classes are in the last axis
    G_of_a = a.logsumexp(axis=1)
    G_of_b = b.logsumexp(axis=1)
    grad_G_of_b = b.softmax(axis=1)
    return G_of_a - G_of_b - inner_product(grad_G_of_b, a - b)
```

### Numpy
