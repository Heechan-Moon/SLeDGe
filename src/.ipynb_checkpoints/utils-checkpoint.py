import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import MinMaxScaler

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
import pandas as pd

import pickle

from typing import Union
from torch_sparse import SparseTensor, matmul

from ucimlrepo import fetch_ucirepo

def dataloader_online(data_name, batch_size=1, num_worker=8):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * x - 1)
    ])
    
    if data_name == "MNIST":
        trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform)
    elif data_name == "KMNIST":
        trainset = torchvision.datasets.KMNIST(root='data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.KMNIST(root='data', train=False, download=True, transform=transform)
    elif data_name == "FashionMNIST":
        trainset = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
    elif data_name == "CIFAR-10":
        trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
    ########################################################
    elif data_name == "Shuttle":
        statlog_shuttle = fetch_ucirepo(id=148)
        X = statlog_shuttle.data.features 
        y = statlog_shuttle.data.targets
        y = y-1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)
    elif data_name == "SLDD":
        data_SLDD = pd.read_csv('data/SLDD/SLDD.txt', sep=r'\s+', header=None)
        X = data_SLDD.iloc[:, :-1].values 
        y = data_SLDD.iloc[:, -1].values
        y = y-1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
    elif data_name == "HAR":
        X_train = pd.read_csv('data/HAR/X_train.txt', sep=r'\s+', header=None)
        y_train = pd.read_csv('data/HAR/y_train.txt', sep=r'\s+', header=None)
        X_test = pd.read_csv('data/HAR/X_test.txt', sep=r'\s+', header=None)
        y_test = pd.read_csv('data/HAR/y_test.txt', sep=r'\s+', header=None)
        X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
        y = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
        y = y-1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)
    elif data_name == "GSAD":
        total_class = []
        total_feature = []
        for k in [1,2,3,4,5,6,7,8,9,10]:
            data_raw = pd.read_csv(f'data/GSAD/batch{k}.dat', header=None)
            for j in range(data_raw.shape[0]):
                value = data_raw.iloc[j,0]
                token = value.split(" ")
                class_num = int(token[0])
                feature = []
                for i in range(1, len(token)):
                    feature.append(float(token[i].split(":")[-1]))
                if len(feature)!=128:
                    print("ERROR")
                total_class.append(class_num)
                total_feature.append(feature)
        X = torch.tensor(total_feature)
        y = torch.tensor(total_class)
        y_tensor = y-1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    elif data_name == "OD":
        occupancy_detection = fetch_ucirepo(id=357)
        X = occupancy_detection.data.features.to_numpy()[:, 1:]
        y = occupancy_detection.data.targets.to_numpy() 
        new_X, new_y = [], []
        for i in range(len(X)):
            valid = True
            for element in X[i]:
                if any(char.isalpha() for char in element):
                    valid = False
            if valid:
                new_X.append(X[i])
                new_y.append(y[i])
        X = np.array(new_X).astype(float)
        y = np.array(new_y)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long).view(-1)
    ########################################################
    elif data_name == "MNDS":
        with open("data/MNDS/MNDS_dataset.pkl", "rb") as f:
            data = pickle.load(f)
        X_tensor = data["X_tensor"]
        y_tensor = data["y_tensor"]
    elif data_name == "Shopper":
        online_shoppers_purchasing_intention_dataset = fetch_ucirepo(id=468)  
        X = online_shoppers_purchasing_intention_dataset.data.features 
        y = online_shoppers_purchasing_intention_dataset.data.targets 
        X_new = X.drop(columns=['Month', 'VisitorType'])
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X_new)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)
    elif data_name == "WebKB":
        with open("data/WebKB/webkb_dataset.pkl", "rb") as f:
            data = pickle.load(f)
        X_tensor = data["X_tensor"]
        y_tensor = data["y_tensor"]
    else:
        raise ValueError("Unsupported dataset: {}".format(data_name))
    
    
    if data_name in ["MNIST", "CIFAR-10", "KMNIST", "FashionMNIST"]:
        full_dataset = ConcatDataset([trainset, testset]) # Concatenate train and test datasets
        dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    else:
        full_dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    
    return dataloader, len(dataloader)

def dataset_adaptation(dataset_name):
    if dataset_name in ["MNIST", "KMNIST", "FashionMNIST"]:
        n_input = 28 * 28
        n_classes = 10
    elif dataset_name in ["CIFAR-10"]:
        n_input = 32 * 32 * 3
        n_classes = 10
    ########################################################
    elif dataset_name == "Shuttle":
        n_input = 7
        n_classes = 7
    elif dataset_name == "SDD":
        n_input = 48
        n_classes = 11
    elif dataset_name == "HAR":
        n_input = 561
        n_classes = 6
    elif dataset_name == "GSAD":
        n_input = 128
        n_classes = 6
    elif dataset_name == "OD":
        n_input = 5
        n_classes = 2
    ########################################################
    elif dataset_name == "MNDS":
        n_input = 500
        n_classes = 17
    elif dataset_name == "Shopper":
        n_input = 15
        n_classes = 2
    elif dataset_name == "WebKB":
        n_input = 500
        n_classes = 7
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
    def __init__(self, gcn_layer, n_input, n_classes, hidden):
        #super().__init__()
        super(GCN_GSL, self).__init__()
        self.n_input, self.n_classes = n_input, n_classes
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.n_hidden = hidden

        self.convs.append(GraphConvolutionLayer(self.n_input, self.n_hidden, bias=True))
        self.bns.append(nn.LayerNorm(self.n_hidden))

        if gcn_layer != 1:
            for _ in range(gcn_layer-1):
                self.convs.append(GraphConvolutionLayer(self.n_hidden, self.n_hidden, bias=True))
                self.bns.append(nn.LayerNorm(self.n_hidden))
            self.convs.append(GraphConvolutionLayer(self.n_hidden, self.n_classes, bias=True))
        
        self.activation = F.relu
        print("="*300)

    def forward(self, in_feat, adj):
        h = in_feat
        for i in range(len(self.convs) - 1):
            h = self.convs[i](h, adj)
            h = self.bns[i](h)
            h = self.activation(h)
        result = self.convs[-1](h, adj)
        return result

    def get_embedding(self, in_feat, adj):
        h = in_feat
        for i in range(len(self.convs) - 1):
            h = self.convs[i](h, adj)
            h = self.bns[i](h)
            h = self.activation(h)
        return h


from embedding_function import MLP, MLP_Light
from pruning_function import kNN

class model_SLeDGe(nn.Module):
    def __init__(self, gcn_layer, n_input, n_classes, scorer_layer, k):
        super(model_SLeDGe, self).__init__()
        self.hidden = 256
        self.gcn = GCN_GSL(gcn_layer, n_input, n_classes, self.hidden)
        self.embedding_function = MLP(scorer_layer, n_input, self.hidden)
        self.pruning_function = kNN(k)
        self.k = k

    def get_adj(self, h):
        Adj_ = self.embedding_function(h)

        valid_k = min(self.k+1, Adj_.shape[0])
        degree = valid_k*torch.ones((Adj_.shape[0])).to(h.device)
        deg_inv_sqrt = degree.pow(-0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        Adj = self.pruning_function(Adj_)
        
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

class model_SLeDGe_Light(nn.Module):
    def __init__(self, gcn_layer, n_input, n_classes, scorer_layer, k):
        super(model_SLeDGe_Light, self).__init__()
        self.hidden = 256
        self.gcn = GCN_GSL(gcn_layer, n_input, n_classes, self.hidden)
        self.embedding_function = MLP_Light(scorer_layer, n_input, self.hidden, k)
        self.pruning_function = kNN(k)
        self.k = k

    def get_adj(self, h, target_index):
        Adj_ = self.embedding_function(h, target_index)

        valid_k = min(self.k+1, Adj_.shape[0])
        degree = valid_k*torch.ones((Adj_.shape[0])).to(h.device)
        deg_inv_sqrt = degree.pow(-0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        adj = D_inv_sqrt @ Adj_ @ D_inv_sqrt
        return adj
        
    def reg(self, features, target_index):
        return self.embedding_function.reg(features, target_index)

    def forward(self, features, target_index):
        Adj = self.get_adj(features, target_index)
        preds = self.gcn(features, Adj)
        return preds

        
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
        "--dataset", 
        type=str, 
        default="MNIST", 
        help="Dataset name"
    )
    parser.add_argument(
        "--label-ratio", 
        type=float, 
        default=0.01, 
        help="The number of labeled element"
    )
    parser.add_argument(
        "--time", 
        action="store_true",  # This makes it a flag that defaults to False
        help="If specified, the tau is 1"
    )

    return parser.parse_args()


def build_labels_from_single_batch_loader(loader, dataset_size, label_ratio, num_classes, device="cpu"):
    LABELS = torch.zeros(dataset_size, dtype=torch.bool, device=device)
    
    first_occurrence = {}
    for idx, (inputs, label, *rest) in enumerate(loader):
        label = int(label.item())

        if label not in first_occurrence:
            first_occurrence[label] = idx

        if len(first_occurrence) == num_classes:
            break
    print(first_occurrence)
    if len(first_occurrence) < num_classes:
        print("Error")

    for idx in first_occurrence.values():
        LABELS[idx] = True

    min_required = len(first_occurrence)
    target_true = max(min_required, int(label_ratio * dataset_size))

    if target_true > dataset_size:
        print("Error")

    current_true = LABELS.sum().item()
    remaining = target_true - current_true

    if remaining > 0:
        candidate_indices = torch.where(~LABELS)[0]
        perm = torch.randperm(len(candidate_indices))
        extra_indices = candidate_indices[perm[:remaining]]
        LABELS[extra_indices] = True

    return LABELS