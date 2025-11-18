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

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader

from tqdm import tqdm
import pandas as pd

from functional import symmetry, normalize, enn, knn

import pickle

def dataloader_online(data_name, batch_size=1, num_worker=8):
    if data_name in ["Caltech101", "Caltech256", "OxfordPets", "Flowers102"]:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2 * x - 1)  # Normalize to [-1, 1]
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
            transforms.Lambda(lambda x: 2 * x - 1)  # Scale from [0, 1] to [-1, 1]
        ])
    
    if data_name == "MNIST":
        trainset = torchvision.datasets.MNIST(root='../data/heechan/data (SSL on Stream)', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='../data/heechan/data (SSL on Stream)', train=False, download=True, transform=transform)
    elif data_name == "EMNIST":
        trainset = torchvision.datasets.EMNIST(root='../data/heechan/data (SSL on Stream)', split='balanced', train=True, download=True, transform=transform)
        testset = torchvision.datasets.EMNIST(root='../data/heechan/data (SSL on Stream)', split='balanced',train=False, download=True, transform=transform)
    elif data_name == "KMNIST":
        trainset = torchvision.datasets.KMNIST(root='../data/heechan/data (SSL on Stream)', train=True, download=True, transform=transform)
        testset = torchvision.datasets.KMNIST(root='../data/heechan/data (SSL on Stream)', train=False, download=True, transform=transform)
    elif data_name == "FashionMNIST":
        trainset = torchvision.datasets.FashionMNIST(root='../data/heechan/data (SSL on Stream)', train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='../data/heechan/data (SSL on Stream)', train=False, download=True, transform=transform)
    elif data_name == "CIFAR-10":
        trainset = torchvision.datasets.CIFAR10(root='../data/heechan/data (SSL on Stream)', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='../data/heechan/data (SSL on Stream)', train=False, download=True, transform=transform)
    elif data_name == "SVHN":
        trainset = torchvision.datasets.SVHN(root='../data/heechan/data (SSL on Stream)', split='train', download=True, transform=transform)
        testset = torchvision.datasets.SVHN(root='../data/heechan/data (SSL on Stream)', split='test', download=True, transform=transform)
    #########
    elif data_name == "CIFAR-100":
        trainset = torchvision.datasets.CIFAR100(root='../data/heechan/data (SSL on Stream)', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='../data/heechan/data (SSL on Stream)', train=False, download=True, transform=transform)
    elif data_name == "STL10":
        trainset = torchvision.datasets.STL10(root='../data/heechan/data (SSL on Stream)', split='train', download=True, transform=transform)
        testset = torchvision.datasets.STL10(root='../data/heechan/data (SSL on Stream)', split='test', download=True, transform=transform)
    elif data_name == "Caltech101": # No official train/test split
        from torchvision import datasets
        trainset = datasets.ImageFolder(root='../data/heechan/data (SSL on Stream)/caltech101/101_ObjectCategories', transform=transform)
    elif data_name == "Caltech256": # No official train/test split
        from torchvision import datasets
        trainset = datasets.ImageFolder(root='../data/heechan/data (SSL on Stream)/caltech256/256_ObjectCategories', transform=transform)
    elif data_name == "Flowers102":
        trainset = torchvision.datasets.Flowers102(root='../data/heechan/data (SSL on Stream)', split='train', download=True, transform=transform)
        valset = torchvision.datasets.Flowers102(root='../data/heechan/data (SSL on Stream)', split='val', download=True, transform=transform)
        testset = torchvision.datasets.Flowers102(root='../data/heechan/data (SSL on Stream)', split='test', download=True, transform=transform)
    elif data_name == "OxfordPets":
        trainset = torchvision.datasets.OxfordIIITPet(root='../data/heechan/data (SSL on Stream)', split='trainval', download=True, target_types='category', transform=transform)
        testset = torchvision.datasets.OxfordIIITPet(root='../data/heechan/data (SSL on Stream)', split='test', download=True, target_types='category', transform=transform)
    elif data_name == "OxfordPets_binary":
        trainset = torchvision.datasets.OxfordIIITPet(root='../data/heechan/data (SSL on Stream)', split='trainval', download=True, target_types='binary-category', transform=transform)
        testset = torchvision.datasets.OxfordIIITPet(root='../data/heechan/data (SSL on Stream)', split='test', download=True, target_types='binary-category', transform=transform)
    #########
    elif data_name == "Camelyon17":
        dataset = get_dataset(dataset="camelyon17", download=True, root_dir="../data/heechan/data (SSL on Stream)")
    
        trainset_full = dataset.get_subset('train', transform=transform)
        valset_full = dataset.get_subset('val', transform=transform)
        id_valset_full = dataset.get_subset('id_val', transform=transform)
        testset_full = dataset.get_subset('test', transform=transform)

        step_1 = 20
        step_2 = 4
        train_indices = list(range(0, len(trainset_full), step_1))
        val_indices = list(range(0, len(valset_full), step_2))
        id_val_indices = list(range(0, len(id_valset_full), step_2))
        test_indices = list(range(0, len(testset_full), step_2))
        print(len(train_indices)+len(val_indices)+len(id_val_indices)+len(test_indices), len(train_indices), len(val_indices), len(id_val_indices), len(test_indices))

        from torch.utils.data import Subset   
        trainset = Subset(trainset_full, train_indices)
        valset = Subset(valset_full, val_indices)
        id_valset = Subset(id_valset_full, id_val_indices)
        testset = Subset(testset_full, test_indices)
    elif data_name == "FMOW":
        dataset = get_dataset(dataset="fmow", download=True, root_dir="../data/heechan/data (SSL on Stream)")
    
        trainset = dataset.get_subset('train', transform=transform)
        id_valset = dataset.get_subset('id_val', transform=transform)
        id_testset = dataset.get_subset('id_test', transform=transform)
        valset = dataset.get_subset('val', transform=transform)
        testset = dataset.get_subset('test', transform=transform)
    elif data_name == "RXRX1":
        dataset = get_dataset(dataset="rxrx1", download=True, root_dir="../data/heechan/data (SSL on Stream)")
    
        trainset = dataset.get_subset('train', transform=transform)
        id_testset = dataset.get_subset('id_test', transform=transform)
        valset = dataset.get_subset('val', transform=transform)
        testset = dataset.get_subset('test', transform=transform)
    elif data_name == "IWILDCAM":
        dataset = get_dataset(dataset="iwildcam", download=True, root_dir="../data/heechan/data (SSL on Stream)")
    
        trainset = dataset.get_subset('train', transform=transform)
        id_valset = dataset.get_subset('id_val', transform=transform)
        id_testset = dataset.get_subset('id_test', transform=transform)
        valset = dataset.get_subset('val', transform=transform)
        testset = dataset.get_subset('test', transform=transform)
    elif data_name == "POVERTY":
        dataset = get_dataset(dataset="poverty", download=True, root_dir="../data/heechan/data (SSL on Stream)")
    
        trainset = dataset.get_subset('train', transform=transform)
        id_valset = dataset.get_subset('id_val', transform=transform)
        id_testset = dataset.get_subset('id_test', transform=transform)
        valset = dataset.get_subset('val', transform=transform)
        testset = dataset.get_subset('test', transform=transform)
    elif data_name == "Shuttle":
        statlog_shuttle = fetch_ucirepo(id=148)
        X = statlog_shuttle.data.features 
        y = statlog_shuttle.data.targets
        y = y-1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)
        """
        train = pd.read_csv('../data/heechan/data (SSL on Stream)/Shuttle/shuttle.trn', sep=r'\s+', header=None)
        X_train = train.iloc[:, -8:-1].values  # Features
        y_train = train.iloc[:, -1].values   # Class
        test = pd.read_csv('../data/heechan/data (SSL on Stream)/Shuttle/shuttle.tst', sep=r'\s+', header=None)
        X_test = test.iloc[:, -8:-1].values  # Features
        y_test = test.iloc[:, -1].values   # Class
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        y = y-1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        """
    elif data_name == "SLDD":
        data_SLDD = pd.read_csv('../data/heechan/data (SSL on Stream)/SLDD/SLDD.txt', sep=r'\s+', header=None)
        X = data_SLDD.iloc[:, :-1].values  # Features
        y = data_SLDD.iloc[:, -1].values   # Class
        y = y-1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
    elif data_name == "HAR":
        X_train = pd.read_csv('../data/heechan/data (SSL on Stream)/HAR/X_train.txt', sep=r'\s+', header=None)
        y_train = pd.read_csv('../data/heechan/data (SSL on Stream)/HAR/y_train.txt', sep=r'\s+', header=None)
        X_test = pd.read_csv('../data/heechan/data (SSL on Stream)/HAR/X_test.txt', sep=r'\s+', header=None)
        y_test = pd.read_csv('../data/heechan/data (SSL on Stream)/HAR/y_test.txt', sep=r'\s+', header=None)
        X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
        y = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
        y = y-1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)
    elif data_name in ["KDDcup", "KDDcup_light"]:
        if data_name == "KDDcup":
            data_raw = pd.read_csv('../data/heechan/data (SSL on Stream)/KDDcup/kddcup', header=None)
        else:
            data_raw = pd.read_csv('../data/heechan/data (SSL on Stream)/KDDcup/kddcup_light', header=None)
        label_raw = data_raw.iloc[:, -1]
        X = data_raw.select_dtypes(exclude=['object'])
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
        label_type = {}
        for element in label_raw:
            temp = label_type.get(element, -1)
            if temp == -1:
                label_type[element]=len(label_type)
        y = label_raw.map(label_type)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)
    elif data_name == "GSD":
        total_class = []
        total_feature = []
        for k in [1,2,3,4,5,6,7,8,9,10]:
            data_raw = pd.read_csv(f'../data/heechan/data (SSL on Stream)/GSD/batch{k}.dat', header=None)
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
    elif data_name == "Occupancy":
        occupancy_detection = fetch_ucirepo(id=357)
        X = occupancy_detection.data.features 
        y = occupancy_detection.data.targets 
        X = X.to_numpy()
        y = y.to_numpy()
        X = X[:, 1:]
        new_X = []
        new_y = []
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
    elif data_name == "IoT":
        X_tensor = torch.load('../data/heechan/data (SSL on Stream)/IoT/features.pt')
        y_tensor = torch.load('../data/heechan/data (SSL on Stream)/IoT/classes.pt')
    elif data_name in ["CR4", "CRE4V2", "FG2C2D", "GEAR2C2D", "MG2C2D"]:
        X_tensor = torch.load(f'../data/heechan/data (SSL on Stream)/Synthetic/{data_name}/features.pt')
        y_tensor = torch.load(f'../data/heechan/data (SSL on Stream)/Synthetic/{data_name}/classes.pt')
    elif data_name in ["WIKI", "REDDIT", "MOOC"]:
        labels = pd.read_csv(f'../data/heechan/data (SSL on Stream)/{data_name}/labels.csv')
        labels_sorted = labels.sort_values(by="time")
        distinct_nodes = labels_sorted["node"].nunique()
        features = torch.randn((distinct_nodes, 128))
        X, y = [], []
        for i in range(len(labels_sorted)):
            if data_name == "MOOC":
                node = int(labels_sorted["node"].iloc[i])
            else:
                node = int(labels_sorted["node"].iloc[i])-1
            X.append(features[node])
            y.append(int(labels_sorted["label"].iloc[i]))
        X_tensor = torch.stack(X)
        y_tensor = torch.tensor(y)
    elif data_name in ["Email-EU"]:
        labels = pd.read_csv(f'../data/heechan/data (SSL on Stream)/{data_name}/labels.csv')
        labels_sorted = labels.sort_values(by="time")
        distinct_nodes = labels_sorted["node"].nunique()
        features = torch.randn((distinct_nodes, 128))
        node_mapping = {}
        for i in range(len(labels_sorted)):
            node = int(labels_sorted["node"].iloc[i])
            if node_mapping.get(node, -1) == -1: node_mapping[node]=len(node_mapping)
        label_mapping = {}
        for i in range(len(labels_sorted)):
            label = int(labels_sorted["label"].iloc[i])
            if label_mapping.get(label, -1) == -1: label_mapping[label]=len(label_mapping)
        X, y = [], []
        for i in range(len(labels_sorted)):
            temp = int(labels_sorted["node"].iloc[i])
            node = node_mapping[temp]
            X.append(features[node])
            temp = labels_sorted["label"].iloc[i]
            y.append(label_mapping[temp])
        X_tensor = torch.stack(X)
        y_tensor = torch.tensor(y)
    elif data_name in ["GDELT-node", "GDELT-node-random"]:
        labels = pd.read_csv('../data/heechan/data (SSL on Stream)/GDELT_sample/labels.csv')
        labels_sorted = labels.sort_values(by="time")
        distinct_nodes = labels_sorted["node"].nunique()
        
        if "random" in data_name:
            features = torch.randn((distinct_nodes, 128))
        else:
            features = torch.load('../data/heechan/data (SSL on Stream)/GDELT_sample/node_features.pt')
            features = features.to(dtype=torch.float)
            
        node_mapping = {}
        for i in range(len(labels_sorted)):
            node = int(labels_sorted["node"].iloc[i])
            if node_mapping.get(node, -1) == -1: node_mapping[node]=len(node_mapping)
        label_mapping = {}
        for i in range(len(labels_sorted)):
            label = int(labels_sorted["label"].iloc[i])
            if label_mapping.get(label, -1) == -1: label_mapping[label]=len(label_mapping)
        X, y = [], []
        for i in range(len(labels_sorted)):
            temp = int(labels_sorted["node"].iloc[i])
            node = node_mapping[temp]
            if "random" in data_name:
                X.append(features[node])
            else:
                X.append(features[temp])
            temp = labels_sorted["label"].iloc[i]
            y.append(label_mapping[temp])
        X_tensor = torch.stack(X)
        y_tensor = torch.tensor(y)
    elif data_name in ["GDELT-edge", "GDELT-edge-random"]:
        labels_edges = pd.read_csv(f'../data/heechan/data (SSL on Stream)/GDELT_sample/labels_edges.csv')
        labels_edges_sorted = labels_edges.sort_values(by="time")
        if "random" not in data_name:
            features = torch.load('../data/heechan/data (SSL on Stream)/GDELT_sample/edge_features.pt')
            features = features.to(dtype=torch.float)
            sorted_indices = labels_edges_sorted.index
            sorted_features = features[sorted_indices]
        label_mapping = {}
        for i in range(len(labels_edges_sorted)):
            label = int(labels_edges_sorted["labels"].iloc[i])
            if label_mapping.get(label, -1) == -1: label_mapping[label]=len(label_mapping)
        X, y = [], []
        for i in range(len(labels_edges_sorted)):
            if "random" not in data_name:
                X.append(sorted_features[i])
            else:
                X.append(torch.randn(128))
            temp = labels_edges_sorted["labels"].iloc[i]
            y.append(label_mapping[temp])
        X_tensor = torch.stack(X)
        y_tensor = torch.tensor(y)
    ##############################################################################
    elif data_name == "ClickStream" :
        df = pd.read_csv("../data/heechan/data (SSL on Stream)/additonal_datasets/ClickStream/clickstream-data.csv", sep=';')
        df.columns = [c.strip().replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]

        # --- Step 3: Define target column ---
        # The 'order' column usually represents the target:
        #   1 = purchase (buy)
        #   0 = browse (no purchase)
        X = df.drop(columns=['order'])
        y = df['order']
        
        # Convert categorical columns to numeric (label encoding)
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category').cat.codes
        
        # --- Step 6: Scale numeric features ---
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_temp = torch.tensor(y, dtype=torch.long).view(-1)
        y_tensor = y_temp - 1
    elif data_name == "CSIC":
        with open("../data/heechan/data (SSL on Stream)/additonal_datasets/CSIC/csic_dataset.pkl", "rb") as f:
            data = pickle.load(f)
        X_tensor = data["X_tensor"]
        y_tensor = data["y_tensor"]
    elif data_name == "MNDS_level_1":
        with open("../data/heechan/data (SSL on Stream)/additonal_datasets/MNDS/MNDS_dataset_level_1.pkl", "rb") as f:
            data = pickle.load(f)
        X_tensor = data["X_tensor"]
        y_tensor = data["y_tensor"]
    elif data_name == "MNDS_level_2":
        with open("../data/heechan/data (SSL on Stream)/additonal_datasets/MNDS/MNDS_dataset_level_2.pkl", "rb") as f:
            data = pickle.load(f)
        X_tensor = data["X_tensor"]
        y_tensor = data["y_tensor"]
    elif data_name == "online_shopper_intention":
        online_shoppers_purchasing_intention_dataset = fetch_ucirepo(id=468)  
        X = online_shoppers_purchasing_intention_dataset.data.features 
        y = online_shoppers_purchasing_intention_dataset.data.targets 
        X_new = X.drop(columns=['Month', 'VisitorType'])
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X_new)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)
    elif data_name == "payloads":
        with open("../data/heechan/data (SSL on Stream)/additonal_datasets/payloads/payloads.pkl", "rb") as f:
            data = pickle.load(f)
        X_tensor = data["X_tensor"]
        y_tensor = data["y_tensor"]
    elif data_name == "WebKB":
        with open("../data/heechan/data (SSL on Stream)/additonal_datasets/WebKB/webkb_dataset.pkl", "rb") as f:
            data = pickle.load(f)
        X_tensor = data["X_tensor"]
        y_tensor = data["y_tensor"]
    elif data_name == "UNSW_NB15":
        with open("../data/heechan/data (SSL on Stream)/UNSW_NB15/UNSW_NB15_dataset.pkl", "rb") as f:
            data = pickle.load(f)
        X_tensor = data["X_tensor"]
        y_tensor = data["y_tensor"]
    elif data_name == "20News":
        with open("../data/heechan/data (SSL on Stream)/20News/20News_dataset.pkl", "rb") as f:
            data = pickle.load(f)
        X_tensor = data["X_tensor"]
        y_tensor = data["y_tensor"]
    elif data_name == "AGNews":
        with open("../data/heechan/data (SSL on Stream)/AGNews/AGNews_dataset.pkl", "rb") as f:
            data = pickle.load(f)
        X_tensor = data["X_tensor"]
        y_tensor = data["y_tensor"]
    elif data_name == "random_10000":
        with open("../data/heechan/data (SSL on Stream)/Random/random_dataset_10000_10_5.pkl", "rb") as f:
            data = pickle.load(f)
        X_tensor = data["X_tensor"]
        y_tensor = data["y_tensor"]
    elif data_name == "random_50000":
        with open("../data/heechan/data (SSL on Stream)/Random/random_dataset_50000_10_5.pkl", "rb") as f:
            data = pickle.load(f)
        X_tensor = data["X_tensor"]
        y_tensor = data["y_tensor"]
    elif data_name == "random_100000":
        with open("../data/heechan/data (SSL on Stream)/Random/random_dataset_100000_10_5.pkl", "rb") as f:
            data = pickle.load(f)
        X_tensor = data["X_tensor"]
        y_tensor = data["y_tensor"]
    elif data_name == "random_200000":
        with open("../data/heechan/data (SSL on Stream)/Random/random_dataset_200000_10_5.pkl", "rb") as f:
            data = pickle.load(f)
        X_tensor = data["X_tensor"]
        y_tensor = data["y_tensor"]
    elif data_name == "random_300000":
        with open("../data/heechan/data (SSL on Stream)/Random/random_dataset_300000_10_5.pkl", "rb") as f:
            data = pickle.load(f)
        X_tensor = data["X_tensor"]
        y_tensor = data["y_tensor"]
    elif data_name == "random_400000":
        with open("../data/heechan/data (SSL on Stream)/Random/random_dataset_400000_10_5.pkl", "rb") as f:
            data = pickle.load(f)
        X_tensor = data["X_tensor"]
        y_tensor = data["y_tensor"]
    elif data_name == "random_500000":
        with open("../data/heechan/data (SSL on Stream)/Random/random_dataset_500000_10_5.pkl", "rb") as f:
            data = pickle.load(f)
        X_tensor = data["X_tensor"]
        y_tensor = data["y_tensor"]
    elif data_name == "random_600000":
        with open("../data/heechan/data (SSL on Stream)/Random/random_dataset_600000_10_5.pkl", "rb") as f:
            data = pickle.load(f)
        X_tensor = data["X_tensor"]
        y_tensor = data["y_tensor"]
    elif data_name == "random_700000":
        with open("../data/heechan/data (SSL on Stream)/Random/random_dataset_700000_10_5.pkl", "rb") as f:
            data = pickle.load(f)
        X_tensor = data["X_tensor"]
        y_tensor = data["y_tensor"]
    elif data_name == "random_800000":
        with open("../data/heechan/data (SSL on Stream)/Random/random_dataset_800000_10_5.pkl", "rb") as f:
            data = pickle.load(f)
        X_tensor = data["X_tensor"]
        y_tensor = data["y_tensor"]
    elif data_name == "random_900000":
        with open("../data/heechan/data (SSL on Stream)/Random/random_dataset_900000_10_5.pkl", "rb") as f:
            data = pickle.load(f)
        X_tensor = data["X_tensor"]
        y_tensor = data["y_tensor"]
    elif data_name == "random_1000000":
        with open("../data/heechan/data (SSL on Stream)/Random/random_dataset_1000000_10_5.pkl", "rb") as f:
            data = pickle.load(f)
        X_tensor = data["X_tensor"]
        y_tensor = data["y_tensor"]
    else:
        raise ValueError("Unsupported dataset: {}".format(data_name))
    
    # Concatenate train and test datasets
    if data_name in [
        "Shuttle", "SLDD", "HAR", "KDDcup", "KDDcup_light", "GSD", "Occupancy", "IoT",
        "CR4", "CRE4V2", "FG2C2D", "GEAR2C2D", "MG2C2D",
        "MOOC", "WIKI", "REDDIT", "Email-EU", "GDELT-node", "GDELT-edge", "GDELT-node-random", "GDELT-edge-random",
        "ClickStream", "CSIC", "MNDS_level_1", "MNDS_level_2", "online_shopper_intention", "payloads", "WebKB", "UNSW_NB15", "20News", "AGNews", 
        "random_10000", "random_50000", "random_100000", "random_200000", "random_300000", "random_400000", "random_500000", "random_600000", "random_700000", "random_800000", "random_900000", "random_1000000"
    ]:
        full_dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    elif data_name in ["Caltech101","Caltech256"]:
        #dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
        dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    elif data_name in ["Flowers102"]:
        full_dataset = ConcatDataset([trainset, valset, testset])
        #dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
        dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    elif data_name in ["Camelyon17"]:
        full_dataset = ConcatDataset([trainset, id_valset, valset, testset])
        #dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
        dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    elif data_name in ["FMoW", "IWILDCAM", "POVERTY"]:
        full_dataset = ConcatDataset([trainset, id_valset, id_testset, valset, testset])
        #dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
        dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    elif data_name in ["RXRX1"]:
        full_dataset = ConcatDataset([trainset, id_testset, valset, testset])
        #dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
        dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    else:
        full_dataset = ConcatDataset([trainset, testset])
        #dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
        dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    
    return dataloader, len(dataloader)

def dataset_adaptation(dataset_name):
    if dataset_name in ["MNIST", "KMNIST", "FashionMNIST"]:
        n_input = 28 * 28  # 28x28 pixel images
        n_classes = 10
    elif dataset_name == "EMNIST":
        n_input = 28 * 28  # 28x28 pixel images
        n_classes = 47  # "balanced" variant
    elif dataset_name in ["CIFAR-10", "SVHN"]:
        n_input = 32 * 32 * 3  # CIFAR-10 and SVHN input size
        n_classes = 10
    ########################################################
    elif dataset_name == "CIFAR-100":
        n_input = 32 * 32 * 3
        n_classes = 100
    elif dataset_name == "STL10":
        n_input = 96 * 96 * 3
        n_classes = 10
    elif dataset_name == "Caltech101":
        n_input = 224 * 224 * 3  # May vary; assuming standard resized input
        n_classes = 102  # 101 + background
    elif dataset_name == "Caltech256":
        n_input = 224 * 224 * 3
        n_classes = 257  # 256 + clutter
    elif dataset_name == "Flowers102":
        n_input = 224 * 224 * 3
        n_classes = 102
    elif dataset_name == "OxfordPets":
        n_input = 224 * 224 * 3
        n_classes = 37
    ########################################################
    elif dataset_name == "Camelyon17":
        n_input = 96 * 96 * 3  # 96x96 pixel RGB images
        n_classes = 2  # Binary classification
    elif dataset_name == "FMOW":
        n_input = 224 * 224 * 3
        n_classes = 62
    elif dataset_name == "RXRX1":
        n_input = 196608
        n_classes = 1139
    ########################################################
    elif dataset_name == "Shuttle":
        n_input = 7
        n_classes = 7
    elif dataset_name == "SLDD":
        n_input = 48
        n_classes = 11
    elif dataset_name == "HAR":
        n_input = 561
        n_classes = 6
    elif dataset_name in ["KDDcup", "KDDcup_light"]:
        n_input = 38
        n_classes = 23
    elif dataset_name == "GSD":
        n_input = 128
        n_classes = 6
    elif dataset_name == "Occupancy":
        n_input = 5
        n_classes = 2
    elif dataset_name == "IoT":
        n_input = 115
        n_classes = 11
    ########################################################
    elif dataset_name in ["CR4", "CRE4V2", "FG2C2D", "GEAR2C2D", "MG2C2D"]:
        n_input = 2
        if dataset_name in ["CR4", "CRE4V2"]:
            n_classes = 4
        else:
            n_classes = 2
    ########################################################
    elif dataset_name in ["MOOC", "WIKI", "REDDIT"]:
        n_input = 128
        n_classes = 2
    elif dataset_name == "Email-EU":
        n_input = 128
        n_classes = 40
    elif dataset_name == "GDELT-node":
        n_input = 413
        n_classes = 80
    elif dataset_name == "GDELT-edge":
        n_input = 182
        n_classes = 81
    elif dataset_name == "GDELT-node-random":
        n_input = 128
        n_classes = 80
    elif dataset_name == "GDELT-edge-random":
        n_input = 128
        n_classes = 81
    ########################################################
    elif dataset_name == "ClickStream":
        n_input = 13
        n_classes = 195
    elif dataset_name == "CSIC":
        n_input = 500
        n_classes = 2
    elif dataset_name == "MNDS_level_1":
        n_input = 500
        n_classes = 17
    elif dataset_name == "MNDS_level_2":
        n_input = 500
        n_classes = 109
    elif dataset_name == "online_shopper_intention":
        n_input = 15
        n_classes = 2
    elif dataset_name == "payloads":
        n_input = 500
        n_classes = 11
    elif dataset_name == "WebKB":
        n_input = 500
        n_classes = 7
    elif dataset_name == "UNSW_NB15": # torch.Size([257673, 196]) torch.Size([257673])
        n_input = 196
        n_classes = 10
    elif dataset_name == "20News": # torch.Size([18846, 500])
        n_input = 500
        n_classes = 20
    elif dataset_name == "AGNews": # torch.Size([127600, 500])
        n_input = 500
        n_classes = 4
    elif "random" in dataset_name:
        n_input = 10
        n_classes = 5
    else:
        raise ValueError("Unsupported dataset: {}".format(dataset_name))
    return n_input, n_classes

def dataloader_kNN_full(data_name, k, device, direction):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
        transforms.Lambda(lambda x: 2 * x - 1)  # Scale from [0, 1] to [-1, 1]
    ])
    
    if "MNIST" in data_name:
        if data_name == "MNIST":
            trainset = torchvision.datasets.MNIST(root='../data/heechan/data (SSL on Stream)', train=True, download=True, transform=transform)
            testset = torchvision.datasets.MNIST(root='../data/heechan/data (SSL on Stream)', train=False, download=True, transform=transform)
        elif data_name == "EMNIST":
            trainset = torchvision.datasets.EMNIST(root='../data/heechan/data (SSL on Stream)', split='balanced', train=True, download=True, transform=transform)
            testset = torchvision.datasets.EMNIST(root='../data/heechan/data (SSL on Stream)', split='balanced',train=False, download=True, transform=transform)
        elif data_name == "KMNIST":
            trainset = torchvision.datasets.KMNIST(root='../data/heechan/data (SSL on Stream)', train=True, download=True, transform=transform)
            testset = torchvision.datasets.KMNIST(root='../data/heechan/data (SSL on Stream)', train=False, download=True, transform=transform)
        elif data_name == "FashionMNIST":
            trainset = torchvision.datasets.FashionMNIST(root='../data/heechan/data (SSL on Stream)', train=True, download=True, transform=transform)
            testset = torchvision.datasets.FashionMNIST(root='../data/heechan/data (SSL on Stream)', train=False, download=True, transform=transform)
        train_data = trainset.data.view(-1, 28 * 28).float()
        test_data = testset.data.view(-1, 28 * 28).float()

        train_labels = trainset.targets
        test_labels = testset.targets
    elif data_name == "CIFAR-10":
        trainset = torchvision.datasets.CIFAR10(root='../data/heechan/data (SSL on Stream)', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='../data/heechan/data (SSL on Stream)', train=False, download=True, transform=transform)
        train_data = torch.tensor(trainset.data).reshape(-1, 32 * 32 * 3).float()
        test_data = torch.tensor(testset.data).reshape(-1, 32 * 32 * 3).float()
        
        train_labels = torch.tensor(trainset.targets)
        test_labels = torch.tensor(testset.targets)
    
    full_data = torch.cat([train_data, test_data], dim=0) #.to(device)
    full_labels = torch.cat([train_labels, test_labels], dim=0) #.to(device)

    num_nodes = full_data.size(0)

    normalized = F.normalize(full_data, dim=1, p=2)
    similarity = normalized@normalized.T
    _, indices = torch.topk(similarity, k=k+1, largest=True)

    src = torch.arange(num_nodes).repeat_interleave(k+1)
    dst = indices.flatten()

    if direction: # True, directed
        knn_graph = dgl.graph((dst, src), num_nodes=num_nodes) # directed
    else:
        knn_graph = dgl.graph((torch.cat([src, dst]), torch.cat([dst, src])), num_nodes=num_nodes) # undirected
    
    # Convert to an undirected graph
    knn_graph.ndata['feat'] = full_data  # Add node features
    knn_graph.ndata['label'] = full_labels  # Add node labels
    return knn_graph


class SimpleMLP_bn(nn.Module):
    def __init__(self, dataset_name, n_layer=3):
        super().__init__()
        super(SimpleMLP_bn, self).__init__()
        self.n_input, self.n_classes = dataset_adaptation(dataset_name)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.n_hidden = 256
        self.classifier = nn.Linear(self.n_hidden, self.n_classes)
        self.activation = F.relu
        print("="*300)
        self.convs.append(nn.Linear(self.n_input, self.n_hidden))
        self.bns.append(nn.LayerNorm(self.n_hidden))
        for _ in range(n_layer-1):
            self.convs.append(nn.Linear(self.n_hidden, self.n_hidden))
            self.bns.append(nn.LayerNorm(self.n_hidden))
    
    def forward(self, in_feat):
        h = in_feat
        for i in range(len(self.convs)):
            h = self.convs[i](h)
            h = self.bns[i](h)
            h = self.activation(h)
        result = self.classifier(h)
        return F.normalize(result)

    def get_embedding(self, in_feat):
        h = in_feat
        for i in range(len(self.convs)):
            h = self.convs[i](h)
            h = self.bns[i](h)
            h = self.activation(h)
        return h

class Simple(nn.Module):
    def __init__(self, dataset_name):
        super().__init__()
        super(Simple, self).__init__()
        self.n_input, self.n_classes = dataset_adaptation(dataset_name)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.n_hidden = 256
        self.classifier = nn.Linear(self.n_hidden, self.n_classes)
        self.activation = F.relu
        print("="*300)
    
    def forward(self, g, in_feat):
        h = in_feat
        for i in range(len(self.convs)):
            h = self.convs[i](g, h)
            h = self.activation(h)
        result = self.classifier(h)
        return F.normalize(result)

    def get_embedding(self, g, in_feat):
        h = in_feat
        for i in range(len(self.convs)):
            h = self.convs[i](g, h)
            h = self.activation(h)
        return h

class SimpleGCN(Simple):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        #super(SimpleGCN, self).__init__()
        self.convs.append(GraphConv(self.n_input, 256, allow_zero_in_degree=True))
        self.convs.append(GraphConv(256, 256, allow_zero_in_degree=True))
        self.convs.append(GraphConv(256, 256, allow_zero_in_degree=True))
        print("-"*300)

class SimpleGCN_edge_weight(Simple):
    def __init__(self, dataset_name, n_layer=3):
        super().__init__(dataset_name)
        #super(SimpleGCN, self).__init__()
        self.convs.append(GraphConv(self.n_input, self.n_hidden, allow_zero_in_degree=True))
        self.bns.append(nn.LayerNorm(self.n_hidden))
        for _ in range(n_layer-1):
            self.convs.append(GraphConv(self.n_hidden, self.n_hidden, allow_zero_in_degree=True))
            self.bns.append(nn.LayerNorm(self.n_hidden))
        print("*"*300)

    def forward(self, g, in_feat):
        h = in_feat
        for i in range(len(self.convs)):
            h = self.convs[i](g, h, edge_weight=g.edata['weight'])
            h = self.bns[i](h)
            h = self.activation(h)
        result = self.classifier(h)
        return F.normalize(result)

    def get_embedding(self, g, in_feat):
        h = in_feat
        for i in range(len(self.convs)):
            h = self.convs[i](g, h, edge_weight=g.edata['weight'])
            h = self.bns[i](h)
            h = self.activation(h)
        return h

def generate_kNN_batch_opt(adj_values, adj_indices, feature_buffer, new_feature_buffer, k, device, direction, weighted=False):
    node_num = feature_buffer.shape[0]
    batch_num = new_feature_buffer.shape[0]

    prev_feature_buffer = feature_buffer[:-batch_num]

    valid_k = min(node_num, k + 1)
    valid_k_2 = min(batch_num, k + 1)

    ################################################################
    #print("EXISTING")
    if adj_values is not None and adj_indices is not None:
        A = F.normalize(prev_feature_buffer, dim=1, p=2)
        B = F.normalize(new_feature_buffer, dim=1, p=2)
        similarity = A@B.T
        values, indices = torch.topk(similarity, k=valid_k_2, largest=True)
        indices += prev_feature_buffer.shape[0]
    
        adj_values = torch.cat([adj_values, values], dim=1)
        adj_indices = torch.cat([adj_indices, indices], dim=1)
    
        sorted_indices = adj_values.argsort(dim=1, descending=True)
        top_indices = sorted_indices[:, :k+1]
        
        adj_values = torch.gather(adj_values, dim=1, index=top_indices)
        adj_indices = torch.gather(adj_indices, dim=1, index=top_indices)
    ################################################################
    #print("New one")
    A = F.normalize(new_feature_buffer, dim=1, p=2)
    B = F.normalize(feature_buffer, dim=1, p=2)
    similarity = A@B.T
    values, indices = torch.topk(similarity, k=valid_k, largest=True)

    if adj_values is None and adj_indices is None:
        adj_values = values
        adj_indices = indices
    else:
        adj_values = torch.cat([adj_values, values], dim=0)
        adj_indices = torch.cat([adj_indices, indices], dim=0)

    edge_index = torch.nonzero(torch.ones_like(adj_indices), as_tuple=True)
    src = edge_index[0] + (node_num - batch_num) if adj_indices.shape[0] == batch_num else edge_index[0]
    dst = adj_indices[edge_index]

    if direction: # True, directed
        g = dgl.graph((dst, src), device=device, num_nodes=node_num)
    else:
        g = dgl.graph((torch.cat([src, dst]), torch.cat([dst, src])), device=device, num_nodes=node_num)
       # g = dgl.to_simple(g.cpu()).to(device)
    return g, adj_values, adj_indices

def generate_kNN_Setting1(X_combined, k, direction):
    num_features = X_combined.shape[0]
    
    if num_features==1:
        src = torch.tensor([0])
        dst = torch.tensor([0])
        g = dgl.graph((src, dst), device=X_combined.device, num_nodes=1)
    else:
        valid_k = min(num_features, k+1)

        normalized = F.normalize(X_combined, dim=1, p=2)
        similarity = normalized@normalized.T
        _, indices = torch.topk(similarity, k=valid_k, largest=True)
    
        src = torch.arange(num_features, device=X_combined.device).repeat_interleave(valid_k)
        dst = indices.flatten()

        if direction: # True, directed
            g = dgl.graph((dst, src), device=X_combined.device, num_nodes=num_features)
        else:
            g = dgl.graph((torch.cat([src, dst]), torch.cat([dst, src])), device=X_combined.device, num_nodes=num_features)
    return g

def generate_kNN_Setting1_ver2(X_combined, k, direction):
    num_features = X_combined.shape[0]
    
    if num_features==1:
        src = torch.tensor([0])
        dst = torch.tensor([0])
        g = dgl.graph((src, dst), device=X_combined.device, num_nodes=1)
        g.edata['weight'] = torch.tensor([1.0], device=X_combined.device)
    else:
        valid_k = min(num_features, k+1)

        normalized = F.normalize(X_combined, dim=1, p=2)
        similarity = normalized@normalized.T
        sim_values, indices = torch.topk(similarity, k=valid_k, largest=True)
    
        src = torch.arange(num_features, device=X_combined.device).repeat_interleave(valid_k)
        dst = indices.flatten()
        weights = sim_values.flatten()

        if direction: # True, directed
            g = dgl.graph((dst, src), device=X_combined.device, num_nodes=num_features)
            g.edata['weight'] = weights
        else:
            g = dgl.graph((torch.cat([src, dst]), torch.cat([dst, src])), device=X_combined.device, num_nodes=num_features)
            g.edata['weight'] = torch.cat([weights, weights])
    return g

from typing import Union
from torch_sparse import SparseTensor, matmul

class APPNPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=0.1, K=10, bias=True, **kwargs):
        super(APPNPLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.K = K
        self.mlp = nn.Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, x: torch.Tensor, adj: Union[SparseTensor, torch.Tensor]):
        h_0 = self.mlp(x)  # Initial prediction
        h = h_0

        for _ in range(self.K):
            if isinstance(adj, SparseTensor):
                h = matmul(adj, h)
            else:
                h = torch.mm(adj, h)
            h = (1 - self.alpha) * h + self.alpha * h_0

        return h

class APPNP_GSL(nn.Module):
    def __init__(self, dataset_name, alpha, K):
        #super().__init__()
        super(APPNP_GSL, self).__init__()
        self.n_input, self.n_classes = dataset_adaptation(dataset_name)
        self.convs = nn.ModuleList()
        self.n_hidden = 256
        self.convs.append(APPNPLayer(self.n_input, self.n_hidden, alpha=alpha, K=K, bias=True))
        self.convs.append(APPNPLayer(self.n_hidden, self.n_hidden, alpha=alpha, K=K, bias=True))
        self.convs.append(APPNPLayer(self.n_hidden, self.n_hidden, alpha=alpha, K=K, bias=True))
        self.classifier = nn.Linear(self.n_hidden, self.n_classes)
        self.activation = F.relu
        print("="*300)
    
    def forward(self, in_feat, adj):
        h = in_feat
        for i in range(len(self.convs)):
            h = self.convs[i](h, adj)
            h = self.activation(h)
        result = self.classifier(h)
        return F.normalize(result, dim=-1)

    def get_embedding(self, in_feat, adj):
        h = in_feat
        for i in range(len(self.convs)):
            h = self.convs[i](h, adj)
            h = self.activation(h)
        return h

class APPNP_GSL_correct(nn.Module):
    def __init__(self, dataset_name, alpha, K):
        #super().__init__()
        super(APPNP_GSL_correct, self).__init__()
        self.n_input, self.n_classes = dataset_adaptation(dataset_name)
        self.convs = nn.ModuleList()
        self.n_hidden = 256
        #self.convs.append(APPNPLayer(self.n_input, self.n_hidden, alpha=alpha, K=K, bias=True))
        #self.convs.append(APPNPLayer(self.n_hidden, self.n_hidden, alpha=alpha, K=K, bias=True))
        #self.convs.append(APPNPLayer(self.n_hidden, self.n_hidden, alpha=alpha, K=K, bias=True))
        self.convs.append(nn.Linear(self.n_input, self.n_hidden, bias=True))
        self.convs.append(nn.Linear(self.n_hidden, self.n_hidden, bias=True))
        self.convs.append(nn.Linear(self.n_hidden, self.n_hidden, bias=True))
        self.classifier = nn.Linear(self.n_hidden, self.n_classes)
        self.activation = F.relu
        self.alpha = alpha
        self.K = K
        print("="*300)
    
    def forward(self, in_feat, adj):
        h = in_feat
        for i in range(len(self.convs)):
            h = self.convs[i](h)
            h = self.activation(h)
        h_0 = h

        # APPPNP iteration
        for _ in range(self.K):
            if isinstance(adj, SparseTensor):
                h = matmul(adj, h)
            else:
                h = torch.mm(adj, h)
            h = (1 - self.alpha) * h + self.alpha * h_0
        
        result = self.classifier(h)
        return F.normalize(result, dim=-1)

    def get_embedding(self, in_feat, adj):
        h = in_feat
        for i in range(len(self.convs)):
            h = self.convs[i](h)
            h = self.activation(h)
        h_0 = h

        # APPPNP iteration
        for _ in range(self.K):
            if isinstance(adj, SparseTensor):
                h = matmul(adj, h)
            else:
                h = torch.mm(adj, h)
            h = (1 - self.alpha) * h + self.alpha * h_0
        return h

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
    def __init__(self, dataset_name):
        #super().__init__()
        super(GCN_GSL, self).__init__()
        self.n_input, self.n_classes = dataset_adaptation(dataset_name)
        self.convs = nn.ModuleList()
        self.n_hidden = 256
        self.convs.append(GraphConvolutionLayer(self.n_input, self.n_hidden, bias=True))
        self.convs.append(GraphConvolutionLayer(self.n_hidden, self.n_hidden, bias=True))
        self.convs.append(GraphConvolutionLayer(self.n_hidden, self.n_hidden, bias=True))
        self.classifier = nn.Linear(self.n_hidden, self.n_classes)
        self.activation = F.relu
        print("="*300)
    
    def forward(self, in_feat, adj):
        h = in_feat
        for i in range(len(self.convs)):
            h = self.convs[i](h, adj)
            h = self.activation(h)
        result = self.classifier(h)
        return F.normalize(result, dim=-1)

    def get_embedding(self, in_feat, adj):
        h = in_feat
        for i in range(len(self.convs)):
            h = self.convs[i](h, adj)
            h = self.activation(h)
        return h

class Simple_Match(nn.Module):
    def __init__(self, dataset_name, layer):
        #super().__init__()
        super(Simple_Match, self).__init__()
        self.n_input, self.n_classes = dataset_adaptation(dataset_name)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.n_hidden = 256
        if layer == 1:
            self.convs.append(nn.Linear(self.n_input, self.n_hidden, bias=True))
            self.bns.append(nn.LayerNorm(self.n_hidden))
        else:
            self.convs.append(nn.Linear(self.n_input, self.n_hidden, bias=True))
            self.bns.append(nn.LayerNorm(self.n_hidden))
            for _ in range(layer-1):
                self.convs.append(nn.Linear(self.n_hidden, self.n_hidden, bias=True))
                self.bns.append(nn.LayerNorm(self.n_hidden))
                
        self.classifier = nn.Linear(self.n_hidden, self.n_classes)
        self.activation = F.relu
        print("="*300)
    
    def forward(self, in_feat):
        h = in_feat
        for i in range(len(self.convs)):
            h = self.convs[i](h)
            h = self.bns[i](h)
            h = self.activation(h)
        result = self.classifier(h)
        return F.normalize(result, dim=-1)

    def get_embedding(self, in_feat):
        h = in_feat
        for i in range(len(self.convs)):
            h = self.convs[i](h)
            h = self.bns[i](h)
            h = self.activation(h)
        return h

class GCN_GSL_bn(nn.Module):
    def __init__(self, dataset_name, gcn_layer):
        #super().__init__()
        super(GCN_GSL_bn, self).__init__()
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

class Simple_GSL(nn.Module):
    def __init__(self, args):
        super(Simple_GSL, self).__init__()
        
        self.gcn = GCN_GSL(args.dataset)

        self.n_input, self.n_classes = dataset_adaptation(args.dataset)
        if args.edge_scorer == "MLP":
            from edge_scorer import MLP
            self.edge_scorer = MLP(args.edge_scorer_layer, self.n_input, 256)
        elif args.edge_scorer == "FP":
            from edge_scorer import FP
            self.edge_scorer = FP()
        elif args.edge_scorer == "ATT":
            from edge_scorer import ATT
            self.edge_scorer = ATT(args.edge_scorer_layer, self.n_input, 256, 3)
        else:
            print("error")
        
        from sparsifier import kNN, eNN
        if args.sparsifier == "kNN":
            self.sparisifer = kNN(args.k)
        elif args.sparsifier == "eNN":
            self.sparisifer = eNN(args.threshold)
        else:
            self.sparisifer = None

        self.processor =  args.processor

        self.k = args.k

    def get_adj(self, h):
        Adj_ = self.edge_scorer(h)

        degree = (self.k+1)*torch.ones((Adj_.shape[0])).to(h.device)
        deg_inv_sqrt = degree.pow(-0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        if self.sparisifer is not None:
            Adj_ = self.sparisifer(Adj_)
            
        if self.processor == "none":
            Adj = Adj_
        #elif self.processor == "sym":
        #    Adj = symmetry(Adj_)
        elif self.processor == "act":
            Adj = F.relu(Adj_)
        #elif self.processor == "actsym":
        #    Adj = F.relu(Adj_)
        #    Adj = symmetry(Adj_)
        else:
            print("error")
            return

        adj = D_inv_sqrt @ Adj @ D_inv_sqrt
        return adj

    def reg(self, features):
        Adj_ = self.edge_scorer(features)
        non_Adj_ = self.edge_scorer.non_forward(features)
        return nn.MSELoss()(Adj_, non_Adj_)
        
    def forward(self, features):
        Adj = self.get_adj(features)
        preds = self.gcn(features, Adj)
        return preds

class Simple_GSL_bn(nn.Module):
    def __init__(self, args):
        super(Simple_GSL_bn, self).__init__()
        
        self.gcn = GCN_GSL_bn(args.dataset, args.gcn_layer)

        self.n_input, self.n_classes = dataset_adaptation(args.dataset)
        if args.edge_scorer == "MLP":
            from edge_scorer import MLP
            self.edge_scorer = MLP(args.edge_scorer_layer, self.n_input, 256)
        elif args.edge_scorer == "FP":
            from edge_scorer import FP
            self.edge_scorer = FP()
        elif args.edge_scorer == "ATT":
            from edge_scorer import ATT
            self.edge_scorer = ATT(args.edge_scorer_layer, self.n_input, 256, 3)
        else:
            print("error")
        
        from sparsifier import kNN, eNN
        if args.sparsifier == "kNN":
            self.sparisifer = kNN(args.k)
        elif args.sparsifier == "eNN":
            self.sparisifer = eNN(args.threshold)
        else:
            self.sparisifer = None

        self.processor =  args.processor

        self.k = args.k

    def get_adj(self, h):
        Adj_ = self.edge_scorer(h)

        degree = (self.k+1)*torch.ones((Adj_.shape[0])).to(h.device)
        deg_inv_sqrt = degree.pow(-0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        if self.sparisifer is not None:
            Adj_ = self.sparisifer(Adj_)
            
        if self.processor == "none":
            Adj = Adj_
        #elif self.processor == "sym":
        #    Adj = symmetry(Adj_)
        elif self.processor == "act":
            Adj = F.relu(Adj_)
        #elif self.processor == "actsym":
        #    Adj = F.relu(Adj_)
        #    Adj = symmetry(Adj_)
        else:
            print("error")
            return

        adj = D_inv_sqrt @ Adj @ D_inv_sqrt
        return adj

    def reg(self, features):
        Adj_ = self.edge_scorer(features)
        non_Adj_ = self.edge_scorer.non_forward(features)
        return nn.MSELoss()(Adj_, non_Adj_)

    def reg_eye(self, features):
        Adj_ = self.edge_scorer(features)
        non_Adj_ = self.edge_scorer.non_forward(features)
        Adj_.fill_diagonal_(0.0)
        non_Adj_.fill_diagonal_(0.0)
        return nn.MSELoss()(Adj_, non_Adj_)

    def reg_orthogonal(self, features):
        embeddings = self.edge_scorer.internal_forward(features)
        I = torch.eye(256).to(embeddings.device)
        L_ortho = torch.norm(embeddings.T @ embeddings - I, p='fro')**2
        return L_ortho

    def reg_HER(self, features): #Hyperspherical Energy Regularization
        sim_matrix = self.edge_scorer(features)
        sim_matrix.fill_diagonal_(0.0)
        energy = torch.sum(sim_matrix ** 2)
        return energy

    def reg_HER_distance(self, features):
        embeddings = self.edge_scorer.internal_forward(features)
        embeddings = F.normalize(embeddings, p=2, dim=1) # Normalize embeddings to lie on unit hypersphere
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)  # shape (N, N) # Compute pairwise squared distances
        dist_matrix = dist_matrix + torch.eye(embeddings.size(0), device=embeddings.device) * 1e6 # Avoid self-distance division by zero
        energy = torch.sum(1.0 / (dist_matrix ** 2 + 1e-8)) # Compute energy = sum of 1 / distance^2
        return energy

    def reg_MI(self, features):
        N = features.shape[0]
        sim_E = self.edge_scorer(features)
        sim_X = self.edge_scorer.non_forward(features)

        mask = ~torch.eye(sim_X.size(0), dtype=torch.bool, device=sim_X.device)
        pos_X = sim_X[mask].view(N, -1)
        pos_E = sim_E[mask].view(N, -1)

        loss = -F.cosine_similarity(pos_X, pos_E, dim=1).mean()
        return loss
        
    def forward(self, features):
        Adj = self.get_adj(features)
        preds = self.gcn(features, Adj)
        return preds

    def get_embedding(self, features):
        Adj = self.get_adj(features)
        embs = self.gcn.get_embedding(features, Adj)
        return embs

class Simple_GSL_bn_ver2(nn.Module):
    def __init__(self, args):
        super(Simple_GSL_bn_ver2, self).__init__()
        
        self.gcn = GCN_GSL_bn(args.dataset)

        self.n_input, self.n_classes = dataset_adaptation(args.dataset)
        if args.edge_scorer == "MLP":
            from edge_scorer import MLP_bn
            self.edge_scorer = MLP_bn(args.edge_scorer_layer, self.n_input, 256)
        elif args.edge_scorer == "FP":
            from edge_scorer import FP
            self.edge_scorer = FP()
        elif args.edge_scorer == "ATT":
            from edge_scorer import ATT
            self.edge_scorer = ATT(args.edge_scorer_layer, self.n_input, 256, 3)
        else:
            print("error")
        
        from sparsifier import kNN, eNN
        if args.sparsifier == "kNN":
            self.sparisifer = kNN(args.k)
        elif args.sparsifier == "eNN":
            self.sparisifer = eNN(args.threshold)
        else:
            self.sparisifer = None

        self.processor =  args.processor

        self.k = args.k

    def get_adj(self, h):
        Adj_ = self.edge_scorer(h)

        degree = (self.k+1)*torch.ones((Adj_.shape[0])).to(h.device)
        deg_inv_sqrt = degree.pow(-0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        if self.sparisifer is not None:
            Adj_ = self.sparisifer(Adj_)
            
        if self.processor == "none":
            Adj = Adj_
        #elif self.processor == "sym":
        #    Adj = symmetry(Adj_)
        elif self.processor == "act":
            Adj = F.relu(Adj_)
        #elif self.processor == "actsym":
        #    Adj = F.relu(Adj_)
        #    Adj = symmetry(Adj_)
        else:
            print("error")
            return

        adj = D_inv_sqrt @ Adj @ D_inv_sqrt
        return adj

    def reg(self, features):
        Adj_ = self.edge_scorer(features)
        non_Adj_ = self.edge_scorer.non_forward(features)
        return nn.MSELoss()(Adj_, non_Adj_)
        
    def forward(self, features):
        Adj = self.get_adj(features)
        preds = self.gcn(features, Adj)
        return preds


class Simple_GSL_binary(nn.Module):
    def __init__(self, args):
        super(Simple_GSL_binary, self).__init__()
        
        self.gcn = GCN_GSL(args.dataset)

        self.n_input, self.n_classes = dataset_adaptation(args.dataset)
        if args.edge_scorer == "MLP":
            from edge_scorer import MLP
            self.edge_scorer = MLP(args.edge_scorer_layer, self.n_input, 256)
        else:
            print("error")
        
        from sparsifier import kNN_binary #, eNN
        if args.sparsifier == "kNN":
            self.sparisifer = kNN_binary(args.k)
        #elif args.sparsifier == "eNN":
        #    self.sparisifer = eNN(args.threshold)
        else:
            self.sparisifer = None

        self.processor =  args.processor

        self.k = args.k

    def get_adj(self, h):
        Adj_ = self.edge_scorer(h)

        degree = (self.k+1)*torch.ones((Adj_.shape[0])).to(h.device)
        deg_inv_sqrt = degree.pow(-0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        if self.sparisifer is not None:
            Adj_ = self.sparisifer(Adj_)
            
        if self.processor == "none":
            Adj = Adj_
        #elif self.processor == "sym":
        #    Adj = symmetry(Adj_)
        elif self.processor == "act":
            Adj = F.relu(Adj_)
        #elif self.processor == "actsym":
        #    Adj = F.relu(Adj_)
        #    Adj = symmetry(Adj_)
        else:
            print("error")
            return

        adj = D_inv_sqrt @ Adj @ D_inv_sqrt
        return adj

    def reg(self, features):
        sim_emb = self.edge_scorer(features)
        sim_feat = self.edge_scorer.non_forward(features)
        return nn.MSELoss()(sim_emb, sim_feat)
        
    def forward(self, features):
        Adj = self.get_adj(features)
        preds = self.gcn(features, Adj)
        return preds

class Simple_GSL_add(nn.Module):
    def __init__(self, args):
        super(Simple_GSL_add, self).__init__()
        
        self.gcn = GCN_GSL(args.dataset)

        self.n_input, self.n_classes = dataset_adaptation(args.dataset)
        if args.edge_scorer == "MLP":
            from edge_scorer import MLP
            self.edge_scorer = MLP(args.edge_scorer_layer, self.n_input, 256)
        elif args.edge_scorer == "FP":
            from edge_scorer import FP
            self.edge_scorer = FP()
        elif args.edge_scorer == "ATT":
            from edge_scorer import ATT
            self.edge_scorer = ATT(args.edge_scorer_layer, self.n_input, 256, 3)
        else:
            print("error")
        
        from sparsifier import kNN, eNN
        if args.sparsifier == "kNN":
            self.sparisifer = kNN(args.k)
        elif args.sparsifier == "eNN":
            self.sparisifer = eNN(args.threshold)
        else:
            self.sparisifer = None

        self.processor =  args.processor

        self.k = args.k
        self.lamb = args.lamb

    def get_adj(self, h):
        Adj_A = self.edge_scorer(h) # Learn
        Adj_B = self.edge_scorer.non_forward(h) # Similarity

        Adj_ = Adj_B + self.lamb * Adj_A

        degree = (self.k+1)*torch.ones((Adj_.shape[0])).to(h.device)
        deg_inv_sqrt = degree.pow(-0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        if self.sparisifer is not None:
            Adj_ = self.sparisifer(Adj_)
            
        if self.processor == "none":
            Adj = Adj_
        #elif self.processor == "sym":
        #    Adj = symmetry(Adj_)
        elif self.processor == "act":
            Adj = F.relu(Adj_)
        #elif self.processor == "actsym":
        #    Adj = F.relu(Adj_)
        #    Adj = symmetry(Adj_)
        else:
            print("error")
            return

        adj = D_inv_sqrt @ Adj @ D_inv_sqrt
        return adj
        
    def forward(self, features):
        Adj = self.get_adj(features)
        preds = self.gcn(features, Adj)
        return preds

class Simple_GSL_add_ver2(nn.Module):
    def __init__(self, args):
        super(Simple_GSL_add_ver2, self).__init__()
        
        self.gcn = GCN_GSL(args.dataset)

        self.n_input, self.n_classes = dataset_adaptation(args.dataset)
        if args.edge_scorer == "MLP":
            from edge_scorer import MLP_ver2
            self.edge_scorer = MLP_ver2(args.edge_scorer_layer, self.n_input, 256)
        elif args.edge_scorer == "FP":
            from edge_scorer import FP
            self.edge_scorer = FP()
        elif args.edge_scorer == "ATT":
            from edge_scorer import ATT
            self.edge_scorer = ATT(args.edge_scorer_layer, self.n_input, 256, 3)
        else:
            print("error")
        
        from sparsifier import kNN, eNN
        if args.sparsifier == "kNN":
            self.sparisifer = kNN(args.k)
        elif args.sparsifier == "eNN":
            self.sparisifer = eNN(args.threshold)
        else:
            self.sparisifer = None

        self.processor =  args.processor

        self.k = args.k
        #self.lamb = args.lamb

    def get_adj(self, h):
        Adj_A = self.edge_scorer(h) # Learn
        Adj_B = self.edge_scorer.non_forward(h) # Similarity

        #Adj_ = Adj_B + self.lamb * Adj_A
        Adj_ = Adj_B + Adj_A

        degree = (self.k+1)*torch.ones((Adj_.shape[0])).to(h.device)
        deg_inv_sqrt = degree.pow(-0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        if self.sparisifer is not None:
            Adj_ = self.sparisifer(Adj_)
            
        if self.processor == "none":
            Adj = Adj_
        #elif self.processor == "sym":
        #    Adj = symmetry(Adj_)
        elif self.processor == "act":
            Adj = F.relu(Adj_)
        #elif self.processor == "actsym":
        #    Adj = F.relu(Adj_)
        #    Adj = symmetry(Adj_)
        else:
            print("error")
            return

        adj = D_inv_sqrt @ Adj @ D_inv_sqrt
        return adj
        
    def forward(self, features):
        Adj = self.get_adj(features)
        preds = self.gcn(features, Adj)
        return preds

class Simple_GSL_APPNP(nn.Module):
    def __init__(self, args):
        super(Simple_GSL_APPNP, self).__init__()
        
        self.appnp = APPNP_GSL(args.dataset, args.alpha, args.K)

        self.n_input, self.n_classes = dataset_adaptation(args.dataset)
        if args.edge_scorer == "MLP":
            from edge_scorer import MLP
            self.edge_scorer = MLP(args.edge_scorer_layer, self.n_input, 256)
        elif args.edge_scorer == "FP":
            from edge_scorer import FP
            self.edge_scorer = FP()
        elif args.edge_scorer == "ATT":
            from edge_scorer import ATT
            self.edge_scorer = ATT(args.edge_scorer_layer, self.n_input, 256, 3)
        else:
            print("error")
        
        from sparsifier import kNN, eNN
        if args.sparsifier == "kNN":
            self.sparisifer = kNN(args.k)
        elif args.sparsifier == "eNN":
            self.sparisifer = eNN(args.threshold)
        else:
            self.sparisifer = None

        self.processor =  args.processor

        self.k = args.k

    def get_adj(self, h):
        Adj_ = self.edge_scorer(h)

        degree = (self.k+1)*torch.ones((Adj_.shape[0])).to(h.device)
        deg_inv_sqrt = degree.pow(-0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        if self.sparisifer is not None:
            Adj_ = self.sparisifer(Adj_)
            
        if self.processor == "none":
            Adj = Adj_
        #elif self.processor == "sym":
        #    Adj = symmetry(Adj_)
        elif self.processor == "act":
            Adj = F.relu(Adj_)
        #elif self.processor == "actsym":
        #    Adj = F.relu(Adj_)
        #    Adj = symmetry(Adj_)
        else:
            print("error")
            return

        adj = D_inv_sqrt @ Adj @ D_inv_sqrt
        return adj

    def reg(self, features):
        Adj_ = self.edge_scorer(features)
        non_Adj_ = self.edge_scorer.non_forward(features)
        return nn.MSELoss()(Adj_, non_Adj_)
        
    def forward(self, features):
        Adj = self.get_adj(features)
        preds = self.appnp(features, Adj)
        return preds

class Simple_GSL_APPNP_correct(nn.Module):
    def __init__(self, args):
        super(Simple_GSL_APPNP_correct, self).__init__()
        
        self.appnp = APPNP_GSL_correct(args.dataset, args.alpha, args.K)

        self.n_input, self.n_classes = dataset_adaptation(args.dataset)
        if args.edge_scorer == "MLP":
            from edge_scorer import MLP
            self.edge_scorer = MLP(args.edge_scorer_layer, self.n_input, 256)
        elif args.edge_scorer == "FP":
            from edge_scorer import FP
            self.edge_scorer = FP()
        elif args.edge_scorer == "ATT":
            from edge_scorer import ATT
            self.edge_scorer = ATT(args.edge_scorer_layer, self.n_input, 256, 3)
        else:
            print("error")
        
        from sparsifier import kNN, eNN
        if args.sparsifier == "kNN":
            self.sparisifer = kNN(args.k)
        elif args.sparsifier == "eNN":
            self.sparisifer = eNN(args.threshold)
        else:
            self.sparisifer = None

        self.processor =  args.processor

        self.k = args.k

    def get_adj(self, h):
        Adj_ = self.edge_scorer(h)

        degree = (self.k+1)*torch.ones((Adj_.shape[0])).to(h.device)
        deg_inv_sqrt = degree.pow(-0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        if self.sparisifer is not None:
            Adj_ = self.sparisifer(Adj_)
            
        if self.processor == "none":
            Adj = Adj_
        #elif self.processor == "sym":
        #    Adj = symmetry(Adj_)
        elif self.processor == "act":
            Adj = F.relu(Adj_)
        #elif self.processor == "actsym":
        #    Adj = F.relu(Adj_)
        #    Adj = symmetry(Adj_)
        else:
            print("error")
            return

        adj = D_inv_sqrt @ Adj @ D_inv_sqrt
        return adj

    def reg(self, features):
        Adj_ = self.edge_scorer(features)
        non_Adj_ = self.edge_scorer.non_forward(features)
        return nn.MSELoss()(Adj_, non_Adj_)
        
    def forward(self, features):
        Adj = self.get_adj(features)
        preds = self.appnp(features, Adj)
        return preds
        
import argparse
def get_parser_memory_GSL():
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
        help="Encoder Type: MLP, GCN"
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
        "--edge_scorer", 
        type=str, 
        default="MLP", 
        help="MLP, ATT, FP"
    )
    parser.add_argument(
        "--edge_scorer_layer", 
        type=int, 
        default=2, 
        help="the number of layer of edge scorer"
    )
    parser.add_argument(
        "--sparsifier", 
        type=str, 
        default="kNN", 
        help="kNN or eNN"
    )
    parser.add_argument(
        "--processor", 
        type=str, 
        default="none", 
        help="none, act" #"none, sym, act, actsym"
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
        default="StreamKMpp_Cosine", 
        help="StreamKMpp_Cosine, ..."
    )
    parser.add_argument(
        "--labeled_memory_type", 
        type=str, 
        default="Window", 
        help="Window, StreamKMpp_Cosine, ..."
    )
    parser.add_argument(
        "--GSL_type", 
        type=str, 
        default="GSL_reg_ver2_bn", 
        help="GSL_reg_ver2_bn, ..."
    )
    parser.add_argument(
        "--lamb_conf", 
        type=float, 
        default=0.0, 
        help="The lamb for confidence score"
    )
    parser.add_argument(
        "--time_decay_tau", 
        type=int, 
        default=100, 
        help="The time decaying temperature"
    )

    return parser.parse_args()

def get_parser_memory():
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
        help="Encoder Type: MLP, GCN"
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
        "--gcn_layer", 
        type=int, 
        default=3, 
        help="The GCN Layer"
    )
    parser.add_argument(
        "--memory_type", 
        type=str, 
        default="StreamKMpp_Cosine", 
        help="StreamKMpp_Cosine, ..."
    )
    parser.add_argument(
        "--labeled_memory_type", 
        type=str, 
        default="Window", 
        help="Window, StreamKMpp_Cosine, ..."
    )
    parser.add_argument(
        "--time_decay_tau", 
        type=int, 
        default=100, 
        help="The time decaying temperature"
    )

    return parser.parse_args()

def get_parser_Match():
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
        "--method", 
        type=str, 
        default="FixMatch", 
        help="Methods: FixMatch, ReFixMatch"
    )
    parser.add_argument(
        "--label-ratio", 
        type=float, 
        default=0.01, 
        help="The number of labeled element"
    )
    parser.add_argument(
        "--memory", 
        type=int, 
        default=1000, 
        help="The memory size"
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
        "--layer", 
        type=int, 
        default=3, 
        help="The GCN Layer"
    )
    parser.add_argument(
        "--memory_type", 
        type=str, 
        default="StreamKMpp_Cosine", 
        help="StreamKMpp_Cosine, ..."
    )
    parser.add_argument(
        "--labeled_memory_type", 
        type=str, 
        default="Window", 
        help="Window, StreamKMpp_Cosine, ..."
    )
    parser.add_argument(
        "--time_decay_tau", 
        type=int, 
        default=100, 
        help="The time decaying temperature"
    )
    parser.add_argument(
        "--lamb", 
        type=float, 
        default=0.01, 
        help="The lamb for reg."
    )
    parser.add_argument(
        "--aug_weak", 
        type=float, 
        default=0.01, 
        help="The weak augmentation"
    )
    parser.add_argument(
        "--aug_strong", 
        type=float, 
        default=0.05, 
        help="The strong augmentation"
    )
    parser.add_argument(
        "--tau", 
        type=float, 
        default=0.95, 
        help="For mask"
    )
    parser.add_argument(
        "--memory_update_ver2", 
        action="store_true",  # This makes it a flag that defaults to False
        help="If specified, the memory update ver.2"
    )

    return parser.parse_args()