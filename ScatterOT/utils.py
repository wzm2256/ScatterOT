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


def cdist(a: torch.Tensor, b: torch.Tensor, a_batch: torch.Tensor, b_batch: torch.Tensor, squreed: bool=True,
          eps: float=1e-5) -> (torch.Tensor, torch.Tensor, torch.Tensor):
	"""Compute distance between all pairs of a and b that are in the same batch
	Args:
		a: (N, D) tensor
		b: (M, D) tensor
		a_batch: (N,) batch index for each point in a
		b_batch: (M,) batch index for each point in b
	Returns:
		dist: (T) vector of distances
		i: (T) row indices of dist
		j: (T) column indices of dist
	"""

	assert a.shape[0] == a_batch.shape[0], "a and a_batch should have the same number of elements"
	assert b.shape[0] == b_batch.shape[0], "b and b_batch should have the same number of elements"
	assert a.shape[1] == b.shape[1], "a and b should have the same dimensionality"

	a_len = a.shape[0]
	b_len = b.shape[0]

	All_a_index = torch.arange(a_len, device=a.device)
	All_b_index = torch.arange(b_len, device=a.device)
	All_edges = torch.stack(torch.meshgrid(All_a_index, All_b_index, indexing='xy')).flatten(start_dim=1)
	Select_index = a_batch[All_edges[0]] == b_batch[All_edges[1]]
	All_edges = All_edges[:, Select_index]

	a_order = a.index_select(0, All_edges[0]) # (T, D)
	b_order = b.index_select(0, All_edges[1]) # (T, D)

	dist = torch.clip(torch.sum((a_order - b_order) ** 2, dim=1), min=eps) # (T,)
	if not squreed:
		dist = torch.sqrt(dist)

	i = All_edges[0]
	j = All_edges[1]

	return dist, i, j



if __name__ == "__main__":
	a=torch.randn(4, 2)
	b=torch.randn(4, 2)
	a_batch=torch.tensor([0, 0, 1, 1])
	b_batch=torch.tensor([0, 1, 1, 1])

	dist, i, j = cdist(a, b, a_batch, b_batch, True)
	print(dist, i, j)

	dist1 = torch.clip(torch.sum((a[:2, :].unsqueeze(0) - b[:1, :].unsqueeze(1)) ** 2, dim=2), min=1e-5) # (T,)
	print(dist1)