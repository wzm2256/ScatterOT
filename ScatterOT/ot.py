import torch
import warnings
import ScatterOT.utils as utils
from typing import Optional

from torch_scatter.composite import scatter_logsumexp
from torch_scatter import scatter_mean, scatter_sum
import pdb


def sinkhorn_log(
    a: Optional[torch.Tensor],
    b: Optional[torch.Tensor],
    batch_a: Optional[torch.Tensor],
    batch_b: Optional[torch.Tensor],
    M: torch.Tensor,
    M_i: torch.Tensor,
    M_j: torch.Tensor,
    reg: float,
    numItermax: int=1000,
    stopThr: float=1e-6,
    verbose=False,
    warn=True,
    **kwargs,
):
    """
    Sinkhorn algorithm for unbalanced OT.

    Parameters
    ----------
    a : torch.Tensor, optional
        Source histogram. If None, a uniform histogram is used.
    b : torch.Tensor, optional
        Target histogram. If None, a uniform histogram is used.
    batch_a : torch.Tensor
        Batch index for points in `a` (default: None).
    batch_b : torch.Tensor
        Batch index for points in `b` (default: None).
    M : torch.Tensor
        cost matrix between points in `a` and `b`.
    M_i : torch.Tensor
        Row indices for elements in `M`.
    M_j : torch.Tensor
        Column indices for elements in `M`.
    reg : float
        Regularization parameter.
    numItermax : int, optional
        Maximum number of iterations (default: 1000).
    stopThr : float, optional
        Stop threshold (default: 1e-6).
    verbose : bool, optional
        Print information (default: False).
    warn : bool, optional
        Raise warning (default: True).
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    torch.Tensor
        Optimal transport matrix
    """
    if a is None:
        a = torch.ones((batch_a.shape[0],), dtype=M.dtype)
        a = a / utils.scatter_keepsize(a, batch_a, 0, sum=True, keepsize=True)
    if b is None:
        b = torch.ones((batch_b.shape[0],), dtype=M.dtype)
        b = b / utils.scatter_keepsize(b, batch_b, 0, sum=True, keepsize=True)

    # init data
    dim_a = len(a)
    dim_b = b.shape[0]

    Mr = -M / reg

    # we assume that no distances are null except those of the diagonal of
    # distances
    u = torch.zeros((dim_a, ), dtype=M.dtype)
    v = torch.zeros((dim_b, ), dtype=M.dtype)


    def get_logT(u, v):
        return Mr + torch.index_select(u, 0, M_i) + torch.index_select(v, 0, M_j)

    loga = torch.log(a)
    logb = torch.log(b)

    err = 1
    for ii in range(numItermax):

        v = logb - scatter_logsumexp(Mr + torch.index_select(u, 0, M_i), M_j, 0)
        u = loga - scatter_logsumexp(Mr + torch.index_select(v, 0, M_j), M_i, 0)

        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            tmp2 = scatter_sum(torch.exp(get_logT(u, v)), M_j,  0)
            err = torch.norm(tmp2 - b)  # violation of marginal

            if verbose:
                if ii % 200 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(ii, err))
            if err < stopThr:
                break
    else:
        if warn:
            warnings.warn(
                "Sinkhorn did not converge. You might want to "
                "increase the number of iterations `numItermax` "
                "or the regularization parameter `reg`."
            )

    return torch.exp(get_logT(u, v))


if __name__ == "__main__":
    import ot
    import scipy
    import numpy as np
    x1 = torch.randn(4, 3) + 5
    x2 = torch.randn(5, 3) - 5

    y1 = torch.randn(2, 3)
    y2 = torch.randn(7, 3)

    # problem 1
    M1 = torch.cdist(x1, y1) ** 2
    M2 = torch.cdist(x2, y2) ** 2
    eps = 20

    out1 = ot.bregman.sinkhorn_log([], [], M1, eps, stopThr=1e-6)
    out2 = ot.bregman.sinkhorn_log([], [], M2, eps, stopThr=1e-6)

    out0 = scipy.linalg.block_diag(out1.cpu().numpy(), out2.cpu().numpy())

    print('-------------------')
    print('POT results:')
    with np.printoptions(precision=3, suppress=True):
        print(out0)

    # problem 2
    batch_a = torch.tensor([0] * x1.shape[0] + [1] * x2.shape[0])
    batch_b = torch.tensor([0] * y1.shape[0] + [1] * y2.shape[0])
    M, M_i, M_j = utils.cdist(torch.cat([x1, x2], dim=0), torch.cat([y1, y2], dim=0), batch_a, batch_b)
    out = sinkhorn_log(None, None, batch_a, batch_b, M, M_i, M_j, eps)

    out1 = torch.sparse_coo_tensor(torch.stack([M_i, M_j], dim=0), out, (batch_a.shape[0], batch_b.shape[0])).to_dense()
    print('-------------------')
    print('ScatterOT results:')
    with np.printoptions(precision=3, suppress=True):
        print(out1.cpu().numpy())

    print('-------------------')
    print('allclose: ', torch.allclose(out1, torch.tensor(out0), atol=1e-5))