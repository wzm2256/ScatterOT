import torch
import warnings

def sinkhorn_log(
    a,
    b,
    M,
    reg,
    numItermax=1000,
    stopThr=1e-9,
    verbose=False,
    warn=True,
    **kwargs,
):

    if len(a) == 0:
        a = torch.full((M.shape[0],), 1.0 / M.shape[0], dtype=M.dtype)
    if len(b) == 0:
        b = torch.full((M.shape[1],), 1.0 / M.shape[1], dtype=M.dtype)

    # init data
    dim_a = len(a)
    dim_b = b.shape[0]

    Mr = -M / reg

    # we assume that no distances are null except those of the diagonal of
    # distances
    u = torch.zeros((dim_a, ), dtype=M.dtype)
    v = torch.zeros((dim_b, ), dtype=M.dtype)

    def get_logT(u, v):
        return Mr + u[:, None] + v[None, :]

    loga = torch.log(a)
    logb = torch.log(b)

    err = 1
    for ii in range(numItermax):
        v = logb - torch.logsumexp(Mr + u[:, None], 0)
        u = loga - torch.logsumexp(Mr + v[None, :], 1)

        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations

            # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
            tmp2 = torch.sum(torch.exp(get_logT(u, v)), 0)
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
    a=torch.tensor([.5, .5])
    b=torch.tensor([.5, .5])
    M=torch.tensor([[0., 1.], [1., 0.]])
    out = sinkhorn_log(a, b, M, 1)
    print(out)

    