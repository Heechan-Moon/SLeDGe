import torch
import torch.nn as nn
import torch.nn.functional as F

class InnerProduct(nn.Module):
    def __init__(self):
        super(InnerProduct, self).__init__()
        pass

    def forward(self, x, y=None, non_negative=False):
        if y is None:
            y = x
        adj = torch.matmul(x, y.T)
        if non_negative:
            mask = (adj > 0).detach().float()
            adj = adj * mask + 0 * (1 - mask)
        return adj

class MLP(torch.nn.Module):
    def __init__(self, nlayers, isize, hsize):
        super(MLP, self).__init__()

        self.layers = torch.nn.ModuleList()
        if nlayers == 1:
            self.layers.append(torch.nn.Linear(isize, hsize))
        else:
            self.layers.append(torch.nn.Linear(isize, hsize))
            for _ in range(nlayers - 1):
                self.layers.append(torch.nn.Linear(hsize, hsize))

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                h = F.relu(h)
        return h

    def forward(self, features):
        embeddings = self.internal_forward(features)
        embeddings = F.normalize(embeddings, dim=1, p=2)
        similarities = InnerProduct()(embeddings)
        return similarities

    def non_forward(self, features):
        embeddings = features
        embeddings = F.normalize(embeddings, dim=1, p=2)
        similarities = InnerProduct()(embeddings)
        return similarities

class MLP_Light(torch.nn.Module):
    def __init__(self, nlayers, isize, hsize, k):
        super(MLP_Light, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(isize, hsize))
        for _ in range(nlayers - 1):
            self.layers.append(torch.nn.Linear(hsize, hsize))

        self.k = k

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                h = F.relu(h)
        return h

    def reg(self, features, target_indices):
        embeddings = self.internal_forward(features)
        embeddings = F.normalize(embeddings, dim=1, p=2)

        target_embeddings = embeddings[target_indices]
        similarities = torch.matmul(target_embeddings, embeddings.t())

        target_feats = features[target_indices]
        feat_similarities = torch.matmul(target_feats, features.t())

        return nn.MSELoss()(similarities, feat_similarities)

    def forward(self, features, target_indices): 
        embeddings = self.internal_forward(features)
        embeddings = F.normalize(embeddings, dim=1, p=2)  # (N, D)
    
        N, D = embeddings.size()
        B = target_indices.size(0)
        device = embeddings.device
        dtype = embeddings.dtype
        
        # ===== N == 1 : self-loop only =====
        if N == 1:
            indices = torch.tensor([[0], [0]], device=device)
            values = torch.tensor([1.0], device=device, dtype=dtype)
            return torch.sparse_coo_tensor(indices, values, size=(1, 1))
    
        # ===== N > 1 =====
        target_embeddings = embeddings[target_indices]
        similarities = torch.matmul(target_embeddings, embeddings.t())
        valid_k = min(N, self.k+1)
        topk_vals, topk_idx = torch.topk(similarities, k=valid_k, largest=True)
        diff_values = topk_vals

        # Construct Undirected Sparse Tensor
        rows = target_indices.repeat_interleave(valid_k).to(device)
        cols = topk_idx.reshape(-1)

        indices = torch.stack([rows, cols], dim=0)
        sparse_vals = diff_values.reshape(-1)
    
        adjacency = torch.sparse_coo_tensor(
            indices, sparse_vals, size=(N, N)
        )

        return adjacency