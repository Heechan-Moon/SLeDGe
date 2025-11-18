import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset, TensorDataset, Subset

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import MinMaxScaler

import torchvision
import torchvision.transforms as transforms

import dgl
from dgl.nn import GraphConv, SAGEConv, GATConv, TAGConv

from tqdm import tqdm
import pandas as pd

import pickle

from typing import Union
from torch_sparse import SparseTensor, matmul

def dataloader_online(data_name, batch_size=1, num_worker=8):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * x - 1)
    ])
    
    if data_name == "MNIST":
        trainset = torchvision.datasets.MNIST(root='/data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='/data', train=False, download=True, transform=transform)
    elif data_name == "KMNIST":
        trainset = torchvision.datasets.KMNIST(root='/data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.KMNIST(root='/data', train=False, download=True, transform=transform)
    elif data_name == "FashionMNIST":
        trainset = torchvision.datasets.FashionMNIST(root='/data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='/data', train=False, download=True, transform=transform)
    elif data_name == "CIFAR-10":
        trainset = torchvision.datasets.CIFAR10(root='/data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='/data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset: {}".format(data_name))
    
    # Concatenate train and test datasets
    full_dataset = ConcatDataset([trainset, testset])
    dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    
    return dataloader, len(dataloader)

def dataset_adaptation(dataset_name):
    if dataset_name in ["MNIST", "KMNIST", "FashionMNIST"]:
        n_input = 28 * 28  # 28x28 pixel images
        n_classes = 10
    elif dataset_name in ["CIFAR-10"]:
        n_input = 32 * 32 * 3
        n_classes = 10
    else:
        raise ValueError("Unsupported dataset: {}".format(dataset_name))
    return n_input, n_classes


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super(GraphConvolutionLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp = nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, x: torch.Tensor, adj: Union[SparseTensor, torch.Tensor]):
        x = self.mlp(x)

        if isinstance(adj, SparseTensor):
            x = matmul(adj, x)
        elif isinstance(adj, torch.Tensor):
            x = torch.mm(adj, x)
        return x
   

class GCN_GSL(nn.Module):
    def __init__(self, dataset_name, gcn_layer):
        #super().__init__()
        super(GCN_GSL, self).__init__()
        self.n_input, self.n_classes = dataset_adaptation(dataset_name)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.n_hidden = 256
        if gcn_layer == 1:
            self.convs.append(GraphConvolutionLayer(self.n_input, self.n_hidden, bias=True))
            self.bns.append(nn.LayerNorm(self.n_hidden))
        else:
            self.convs.append(GraphConvolutionLayer(self.n_input, self.n_hidden, bias=True))
            self.bns.append(nn.LayerNorm(self.n_hidden))
            for _ in range(gcn_layer-1):
                self.convs.append(GraphConvolutionLayer(self.n_hidden, self.n_hidden, bias=True))
                self.bns.append(nn.LayerNorm(self.n_hidden))
                
        self.classifier = nn.Linear(self.n_hidden, self.n_classes)
        self.activation = F.relu
        print("="*300)
    
    def forward(self, in_feat, adj):
        h = in_feat
        for i in range(len(self.convs)):
            h = self.convs[i](h, adj)
            h = self.bns[i](h)
            h = self.activation(h)
        result = self.classifier(h)
        return F.normalize(result, dim=-1)

    def get_embedding(self, in_feat, adj):
        h = in_feat
        for i in range(len(self.convs)):
            h = self.convs[i](h, adj)
            h = self.bns[i](h)
            h = self.activation(h)
        return h


from embedding_function import MLP
class Simple_GSL(nn.Module):
    def __init__(self, args):
        super(Simple_GSL, self).__init__()
        
        self.gcn = GCN_GSL(args.dataset, args.gcn_layer)

        self.n_input, self.n_classes = dataset_adaptation(args.dataset)
        if args.embedding_function == "MLP":
            self.embedding_function = MLP(args.embedding_function_layer, self.n_input, 256)
        else:
            print("error")
        
        from pruning_function import kNN
        if args.pruning_function == "kNN":
            self.pruning_function = kNN(args.k)
        else:
            self.pruning_function = None

        self.processor =  args.processor

        self.k = args.k

    def get_adj(self, h):
        Adj_ = self.embedding_function(h)

        degree = (self.k+1)*torch.ones((Adj_.shape[0])).to(h.device)
        deg_inv_sqrt = degree.pow(-0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        if self.pruning_function is not None:
            Adj_ = self.pruning_function(Adj_)
            
        if self.processor == "none":
            Adj = Adj_
        elif self.processor == "act":
            Adj = F.relu(Adj_)
        else:
            print("error")
            return

        adj = D_inv_sqrt @ Adj @ D_inv_sqrt
        return adj

    def reg(self, features):
        Adj_ = self.embedding_function(features)
        non_Adj_ = self.embedding_function.non_forward(features)
        return nn.MSELoss()(Adj_, non_Adj_)
        
    def forward(self, features):
        Adj = self.get_adj(features)
        preds = self.gcn(features, Adj)
        return preds

    def get_embedding(self, features):
        Adj = self.get_adj(features)
        embs = self.gcn.get_embedding(features, Adj)
        return embs

        
import argparse
def get_parser():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Specify GPU number for PyTorch training.")
    parser.add_argument(
        "--gpu", 
        type=int, 
        default=0, 
        help="GPU number to use (default: 0). If no GPU is available, CPU will be used."
    )
    parser.add_argument(
        "--opt", 
        type=str, 
        default="Adam",  # "Adagrad"
        help="Optimizer to use"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="MNIST", 
        help="Dataset name"
    )
    parser.add_argument(
        "--encoder", 
        type=str, 
        default="GCN", 
        help="Encoder Type: GCN"
    )
    parser.add_argument(
        "--label-ratio", 
        type=float, 
        default=0.01, 
        help="The number of labeled element"
    )
    parser.add_argument(
        "--k", 
        type=int, 
        default=1, 
        help="The k nearest neighborhood"
    )
    parser.add_argument(
        "--memory", 
        type=int, 
        default=1000, 
        help="The memory size"
    )
    parser.add_argument(
        "--directed", 
        action="store_true",  # This makes it a flag that defaults to False
        help="If specified, the graph will be directed"
    )
    parser.add_argument(
        "--memory_constant", 
        action="store_true",  # This makes it a flag that defaults to False
        help="If specified, the size of memory will be constant."
    )
    parser.add_argument(
        "--labeled_size", 
        type=int, 
        default=100, 
        help="the size of label memory"
    )
    parser.add_argument(
        "--embedding_function", 
        type=str, 
        default="MLP", 
        help="MLP"
    )
    parser.add_argument(
        "--embedding_function_layer", 
        type=int, 
        default=1, 
        help="the number of layer of embedding_function"
    )
    parser.add_argument(
        "--pruning_function", 
        type=str, 
        default="kNN", 
        help="kNN"
    )
    parser.add_argument(
        "--processor", 
        type=str, 
        default="none", 
        help="none, act"
    )
    parser.add_argument(
        "--lamb", 
        type=float, 
        default=0.01, 
        help="The lamb for reg."
    )
    parser.add_argument(
        "--gcn_layer", 
        type=int, 
        default=3, 
        help="The GCN Layer"
    )
    parser.add_argument(
        "--memory_type", 
        type=str, 
        default="Update", 
        help="Unlabeld Memory Type"
    )
    parser.add_argument(
        "--labeled_memory_type", 
        type=str, 
        default="Window", 
        help="Window, Window_time_decaying"
    )
    parser.add_argument(
        "--GSL_type", 
        type=str, 
        default="SLeDGe", 
        help="SLeDGe, ..."
    )
    parser.add_argument(
        "--time_decay_tau", 
        type=int, 
        default=100, 
        help="The time decaying temperature"
    )

    return parser.parse_args()


