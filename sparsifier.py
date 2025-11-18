import torch
import torch.nn as nn

#from functional import knn

def knn(adj, K, self_loop=True, set_value=None, sparse_out=False):
    if adj.is_sparse:
        pass
    else:
        device = adj.device
        k = min(adj.size(-1), int(K))
        values, indices = adj.topk(k=k, dim=-1)
        assert torch.max(indices) < adj.shape[1]
        if sparse_out:
            n = adj.shape[0]
            new_indices = torch.stack([torch.arange(n, device=device).view(-1, 1).expand(-1, int(K)).contiguous().flatten(),
                                   indices.flatten()])
            new_values = values.flatten()
            return torch.sparse.FloatTensor(new_indices, new_values, [n, n]).coalesce()
        else:
            mask = torch.zeros(adj.shape, device=device)
            mask[torch.arange(adj.shape[0], device=device).view(-1, 1), indices] = 1.
            if not self_loop:
                mask[torch.arange(adj.shape[0], device=device).view(-1, 1), torch.arange(adj.shape[0], device=device).view(-1, 1)] = 0
            mask.requires_grad = False
            new_adj = adj * mask
            if set_value:
                new_adj[new_adj.nonzero()[:, 0], new_adj.nonzero()[:, 1]] = set_value
            return new_adj

class kNN(torch.nn.Module):
    def __init__(self, k):
        super(kNN, self).__init__()
        self.k = k

    def forward(self, similarities):
        return knn(similarities, self.k + 1)
