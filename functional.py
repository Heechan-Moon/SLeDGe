import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import torch_geometric.nn

def scipy_sparse_to_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_tensor_to_scipy_sparse(sparse_tensor):
    sparse_tensor = sparse_tensor.cpu()
    row = sparse_tensor.coalesce().indices()[0].numpy()
    col = sparse_tensor.coalesce().indices()[1].numpy()
    values = sparse_tensor.coalesce().values().numpy()
    return sp.coo_matrix((values, (row, col)), shape=sparse_tensor.shape)

class Cosine(nn.Module):
    '''
    Cosine to generate pairwise similarities from given node embeddings.
    '''
    def __init__(self):
        super(Cosine, self).__init__()
        pass

    def forward(self, x, y=None, non_negative=False):
        '''
        Given two groups of node embeddings, calculate the pairwise similarities.

        Parameters
        ----------
        x : torch.tensor
            Input features.
        y : torch.tensor
            Input features. ``x`` will be used if set to `None`.
        non_negative : bool
            Whether to mask negative elements.

        Returns
        -------
        adj : torch.tensor
            Pairwise similarities.
        '''
        if y is None:
            y = x
        context_x = F.normalize(x, p=2, dim=-1)
        context_y = F.normalize(y, p=2, dim=-1)
        adj = torch.matmul(context_x, context_y.T)
        if non_negative:
            mask = (adj > 0).detach().float()
            adj = adj * mask + 0 * (1 - mask)
        return adj


def normalize(mx, style='symmetric', add_loop=True, p=None):
    if style == 'row':
        if mx.is_sparse:
            return row_normalize_sp(mx)
        else:
            return row_nomalize(mx)
    elif style == 'symmetric':
        if mx.is_sparse:
            return normalize_sp_tensor_tractable(mx, add_loop)
        else:
            return normalize_tensor(mx, add_loop)
    elif style == 'softmax':
        if mx.is_sparse:
            return torch.sparse.softmax(mx, dim=-1)
        else:
            return F.softmax(mx, dim=-1)
    elif style == 'row-norm':
        assert p is not None
        if mx.is_sparse:
            # TODO
            pass
        else:
            return F.normalize(mx, dim=-1, p=p)
    else:
        raise KeyError("The normalize style is not provided.")


def row_nomalize(mx):

    r_sum = mx.sum(1)
    r_inv = r_sum.pow(-1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx

    return mx


def row_normalize_sp(mx):
    adj = mx.coalesce()
    inv_sqrt_degree = 1. / (torch.sparse.sum(mx, dim=1).values() + 1e-12)
    D_value = inv_sqrt_degree[adj.indices()[0]]
    new_values = adj.values() * D_value
    return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())


def normalize_sp_tensor_tractable(adj, add_loop=True):
    n = adj.shape[0]
    device = adj.device
    if add_loop:
        adj = adj + torch.eye(n, device=device).to_sparse()
    adj = adj.coalesce()
    inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()) + 1e-12)
    D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]
    new_values = adj.values() * D_value
    return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())


def normalize_tensor(adj, add_loop=True):
    device = adj.device
    adj_loop = adj + torch.eye(adj.shape[0], device=device) if add_loop else adj
    rowsum = adj_loop.sum(1)
    r_inv = rowsum.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    A = r_mat_inv @ adj_loop
    A = A @ r_mat_inv
    return A


def normalize_sp_tensor(adj, add_loop=True):
    device = adj.device
    adj = sparse_tensor_to_scipy_sparse(adj)
    adj = normalize_sp_matrix(adj, add_loop)
    adj = scipy_sparse_to_sparse_tensor(adj, device=device)
    return adj


def normalize_sp_matrix(adj, add_loop=True):
    mx = adj + sp.eye(adj.shape[0]) if add_loop else adj
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    new = mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    return new


def symmetry(adj, i=2):
    if adj.is_sparse:
        n = adj.shape[0]
        adj_t = torch.sparse.FloatTensor(adj.indices()[[1,0]], adj.values(), [n, n])
        return (adj_t + adj).coalesce() / i
    else:
        return (adj.t() + adj) / i


def knn(adj, K, self_loop=True, set_value=None, sparse_out=False):
    if adj.is_sparse:
        # TODO
        pass
    else:
        device = adj.device
        k = min(adj.size(-1), int(K))
        values, indices = adj.topk(k=k, dim=-1)
        #values, indices = adj.topk(k=int(K), dim=-1)
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

def knn_ste(adj, K, self_loop=True, set_value=None, sparse_out=False):
    device = adj.device
    k = min(adj.size(-1), int(K))

    # Forward: Hard top-k mask
    values, indices = adj.topk(k=k, dim=-1)
    hard_mask = torch.zeros_like(adj)
    hard_mask[torch.arange(adj.size(0), device=device).view(-1, 1), indices] = 1.0

    if not self_loop:
        hard_mask[torch.arange(adj.shape[0]), torch.arange(adj.shape[0])] = 0.0

    # Backward: Soft version — allow gradients to flow through adj
    soft_mask = hard_mask.detach() - adj.detach() + adj

    # Final adjacency: STE approximation
    new_adj = adj * soft_mask  # Forward uses hard_mask, backward flows through adj

    if set_value is not None:
        nonzero_idx = new_adj.nonzero(as_tuple=False)
        new_adj[nonzero_idx[:, 0], nonzero_idx[:, 1]] = set_value

    if sparse_out:
        n = adj.shape[0]
        indices = new_adj.nonzero(as_tuple=False).T
        values = new_adj[indices[0], indices[1]]
        return torch.sparse.FloatTensor(indices, values, torch.Size([n, n])).coalesce()
    else:
        return new_adj
        
def dilated_knn(adj, K, dilation=1, self_loop=True, set_value=None, sparse_out=False):
    if adj.is_sparse:
        # TODO: Sparse case not implemented
        pass
    else:
        device = adj.device
        max_k = min(adj.size(-1), K * dilation)
        topk_values, topk_indices = adj.topk(k=max_k, dim=-1)

        # Apply dilation: take every dilation-th neighbor
        dilated_indices = topk_indices[:, ::dilation]
        dilated_values = topk_values[:, ::dilation]

        assert torch.max(dilated_indices) < adj.shape[1]

        if sparse_out:
            n = adj.shape[0]
            new_indices = torch.stack([
                torch.arange(n, device=device).view(-1, 1).expand(-1, dilated_indices.size(1)).contiguous().flatten(),
                dilated_indices.flatten()
            ])
            new_values = dilated_values.flatten()
            return torch.sparse.FloatTensor(new_indices, new_values, [n, n]).coalesce()
        else:
            mask = torch.zeros(adj.shape, device=device)
            mask[torch.arange(adj.shape[0], device=device).view(-1, 1), dilated_indices] = 1.
            if not self_loop:
                mask[torch.arange(adj.shape[0], device=device), torch.arange(adj.shape[0], device=device)] = 0
            mask.requires_grad = False
            new_adj = adj * mask
            if set_value is not None:
                new_adj[new_adj.nonzero(as_tuple=True)] = set_value
            return new_adj


def enn(adj, epsilon, set_value=None):
    if adj.is_sparse:
        n = adj.shape[0]
        values = adj.values()
        mask = values > epsilon
        mask.requires_grad = False
        new_values = values[mask]
        if set_value:
            new_values[:] = set_value
        new_indices = adj.indices()[:,mask]
        return torch.sparse.FloatTensor(new_indices, new_values, [n, n])
    else:
        mask = adj > epsilon
        mask.requires_grad = False
        new_adj = adj * mask
        if set_value:
            new_adj[mask] = set_value
        return new_adj


def to_undirected(adj):
    if adj.is_sparse:
        device = adj.device
        assert (adj.values() == 1).all()
        n = adj.shape[0]
        indices_t = adj.indices()[[1,0]]
        new_indices = torch.cat([adj.indices(), indices_t], dim=1)
        new_indices = torch.unique(new_indices, dim=1)
        new_values = torch.ones(new_indices.shape[1], device=device)
        new_adj = torch.sparse.FloatTensor(new_indices, new_values, [n, n])
        return new_adj
    else:
        return adj + adj.T - adj * (adj <= adj.T) - adj.T * (adj > adj.T)


def knn_fast(X, k, b):
    X = F.normalize(X, dim=1, p=2)
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1)).cuda()
    rows = torch.zeros(X.shape[0] * (k + 1)).cuda()
    cols = torch.zeros(X.shape[0] * (k + 1)).cuda()
    norm_row = torch.zeros(X.shape[0]).cuda()
    norm_col = torch.zeros(X.shape[0]).cuda()
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        norm_row[index: end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))
    return rows, cols, values


def apply_non_linearity(adj, non_linearity, i):
    if non_linearity == 'elu':
        return F.elu(adj * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(adj)
    elif non_linearity == 'none':
        return adj
    else:
        raise KeyError('We dont support the non-linearity yet')


def removeselfloop(adj):
    if adj.is_sparse:
        # TODO
        pass
    else:
        n = adj.shape[0]
        mask = torch.eye(n, device=adj.device).bool()
        return adj * ~mask

