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
