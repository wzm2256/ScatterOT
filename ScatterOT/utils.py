from torch_scatter import scatter_mean, scatter_sum, segment_coo
import torch

def scatter_keepsize(X, index, dim, keepsize=False, sum=False):
    """Compute mean of `X` along `dim` using `index`"""
    if sum:
        mean = scatter_sum(X, index, dim=dim)
    else:
        mean = scatter_mean(X, index, dim=dim)
    if not keepsize:
        return mean
    else:
        return torch.index_select(mean, dim, index)

