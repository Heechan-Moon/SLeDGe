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

class DiagonalLinear(torch.nn.Module):
    def __init__(self, size_in, size_out):
        super(DiagonalLinear, self).__init__()
        assert size_in == size_out, "DiagonalLinear requires input and output size to match"
        self.weight = torch.nn.Parameter(torch.ones(size_in))
        self.bias = torch.nn.Parameter(torch.zeros(size_in))

    def forward(self, x):
        return x * self.weight + self.bias

class ATT(torch.nn.Module):
    def __init__(self, nlayers, isize, hsize, nheads):
        super(ATT, self).__init__()

        self.layers = torch.nn.ModuleList()
        for _ in range(nheads):
            temp = torch.nn.ModuleList()
            if nlayers == 1:
                #temp.append(DiagonalLinear(isize, hsize))
                temp.append(torch.nn.Linear(isize, hsize))
            else:
                #temp.append(DiagonalLinear(isize, hsize))
                temp.append(torch.nn.Linear(isize, hsize))
                for _ in range(nlayers - 1):
                    temp.append(DiagonalLinear(hsize, hsize))
            self.layers.append(temp)
            
    def forward(self, features):
        results = []
        for _, temp in enumerate(self.layers):
            h = features
            for i, layer in enumerate(temp):
                h = layer(h)
                if i != (len(temp) - 1):
                    h = F.relu(h)
            embeddings = F.normalize(h, dim=1, p=2)
            similarities = InnerProduct()(embeddings)
            results.append(similarities)
        return torch.mean(torch.stack(results, dim=0), dim=0)

class FP(nn.Module):
    def __init__(self):
        super(FP, self).__init__()
        #self.fp = nn.Parameter(torch.randn(1, 1))
        self.fp = nn.Parameter(torch.ones(1, 1))

    def forward(self, features):
        return self.fp

    def expand(self, node_num):
        if self.fp.shape[0] < node_num:
            prev_node_num = self.fp.shape[0]
    
            #new_fp = torch.randn((node_num, node_num), device=self.fp.device)
            new_fp = torch.ones((node_num, node_num), device=self.fp.device)
            new_fp[:prev_node_num, :prev_node_num] = self.fp.data  # Copy existing values
    
            # Replace the parameter
            self.fp = nn.Parameter(new_fp)
            return True
        else:
            return False