import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

from utils_online_integrated_Setting1 import *
from Memory import *

def SSL_kNN_full(kNN, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    graph = kNN.to(device)
    epochs = 1000

    dataset_size = kNN.num_nodes()
    mask = torch.zeros(dataset_size, dtype=torch.bool).to(device)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    mask[true_indices] = True  # Set selected indices to True
    eval_mask = ~mask
    print(len(mask), torch.sum(mask), mask)
    print(len(eval_mask), torch.sum(eval_mask), eval_mask)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "MLP":
        model = SimpleMLP(dataset_name).to(device)
    elif encoder_name == "GCN":
        model = SimpleGCN(dataset_name).to(device)
    elif encoder_name == "GraphSage":
        model = SimpleGraphSage(dataset_name).to(device)
    elif encoder_name == "GAT":
        model = SimpleGAT(dataset_name).to(device)
    elif encoder_name == "TAG":
        model = SimpleTAG(dataset_name).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    model.train()
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        outputs = model(graph, graph.ndata['feat'])[mask]
        loss = criterion(outputs, graph.ndata['label'][mask].to(device))
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}: Loss - {loss:.4f}")

    model.eval()
    with torch.no_grad():
        outputs = model(graph, graph.ndata['feat'])
        
    preds = torch.argmax(outputs, dim=1)
    accuracy = (preds[eval_mask] == graph.ndata['label'][eval_mask].to(device)).float().mean().item()
                
    return accuracy

def SSL_kNN_semi_full_online_update(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    batch_size = args.batch

    LABELS = torch.zeros(dataset_size, dtype=torch.bool).to(device)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "MLP":
        model = SimpleMLP(dataset_name).to(device)
    elif encoder_name == "GCN":
        model = SimpleGCN(dataset_name).to(device)
    elif encoder_name == "GraphSage":
        model = SimpleGraphSage(dataset_name).to(device)
    elif encoder_name == "GAT":
        model = SimpleGAT(dataset_name).to(device)
    elif encoder_name == "TAG":
        model = SimpleTAG(dataset_name).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer, label_buffer, adj_values, adj_indices = None, None, None, None
    for idx, (inputs, labels, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=len(loader))):
        extra_args = optional if dataset_name == "Camelyon17" else []

        inputs = inputs.to(device)
        labels = labels.to(device)

        size = inputs.shape[0]
        
        if feature_buffer is None:
            feature_buffer = inputs.view(size, -1)
            label_buffer = labels.view(size)
        else: 
            feature_buffer = torch.cat([feature_buffer, inputs.view(size, -1)], dim=0)
            label_buffer = torch.cat([label_buffer, labels.view(size)], dim=0)

        if encoder_name != "MLP":
            graph, adj_values, adj_indices = generate_kNN_batch_opt(adj_values, adj_indices, feature_buffer, inputs.view(size, -1), nearest_k, device, direction=direction)
        
        # --- Test Phase ---
        model.eval()
        with torch.no_grad():
            if encoder_name == "MLP":
                outputs = model(feature_buffer)[-size:]
            else:
                outputs = model(graph, feature_buffer)[-size:]
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == labels).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % (10000//batch_size) == 0:
            print(f"{idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
            print(graph)
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        element_num = feature_buffer.shape[0]
        mask = LABELS[:element_num]
        if encoder_name == "MLP":
            loss = criterion(model(feature_buffer)[mask], label_buffer[mask])
        else:
            loss = criterion(model(graph, feature_buffer)[mask], label_buffer[mask])
        loss.backward()
        optimizer.step()

        #graph = graph.cpu()
        if encoder_name != "MLP":
            del graph
        torch.cuda.empty_cache()
    return test_accuracies

def SSL_online_kNN_online_update_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "MLP":
        model = SimpleMLP(dataset_name).to(device)
    elif encoder_name == "GCN":
        model = SimpleGCN(dataset_name).to(device)
    elif encoder_name == "GraphSage":
        model = SimpleGraphSage(dataset_name).to(device)
    elif encoder_name == "GAT":
        model = SimpleGAT(dataset_name).to(device)
    elif encoder_name == "TAG":
        model = SimpleTAG(dataset_name).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    #feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    feature_buffer_label, labels = torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        feature_buffer_unlabel = torch.empty(0, device=device)

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
        else:
            feature_buffer_unlabel = torch.cat((feature_buffer_unlabel, input))

        X_combined = torch.cat((feature_buffer_label, feature_buffer_unlabel))
    
        if encoder_name == "MLP":
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(X_combined)[-1].view(1, -1)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            graph = generate_kNN_Setting1(X_combined, nearest_k, direction=direction)
            
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(graph, X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(graph, X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0:
            print(f"{idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
            if encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
                print(graph)
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        if encoder_name == "MLP":
            loss = criterion(model(feature)[:len(feature_buffer_label)], labels)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            loss = criterion(model(graph, X_combined)[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = args.memory

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "MLP":
        model = SimpleMLP(dataset_name).to(device)
    elif encoder_name == "GCN":
        model = SimpleGCN(dataset_name).to(device)
    elif encoder_name == "GraphSage":
        model = SimpleGraphSage(dataset_name).to(device)
    elif encoder_name == "GAT":
        model = SimpleGAT(dataset_name).to(device)
    elif encoder_name == "TAG":
        model = SimpleTAG(dataset_name).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
        else:
            feature_buffer_unlabel = torch.cat((feature_buffer_unlabel, input))

        if len(feature_buffer_unlabel)>memory:
            feature_buffer_unlabel=feature_buffer_unlabel[-memory:]

        X_combined = torch.cat((feature_buffer_label, feature_buffer_unlabel))
    
        if encoder_name == "MLP":
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(X_combined)[-1].view(1, -1)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            graph = generate_kNN_Setting1(X_combined, nearest_k, direction=direction)
            
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(graph, X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(graph, X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0:
            print(f"{len(feature_buffer_label)} labeled, {len(feature_buffer_unlabel)} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
            if encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
                print(graph)
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        if encoder_name == "MLP":
            loss = criterion(model(feature)[:len(feature_buffer_label)], labels)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            loss = criterion(model(graph, X_combined)[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_ver2_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = args.memory

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "MLP":
        model = SimpleMLP(dataset_name).to(device)
    elif encoder_name == "GCN":
        model = SimpleGCN_edge_weight(dataset_name, n_layer=args.gcn_layer).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
        else:
            feature_buffer_unlabel = torch.cat((feature_buffer_unlabel, input))

        if len(feature_buffer_unlabel)>memory:
            feature_buffer_unlabel=feature_buffer_unlabel[-memory:]

        X_combined = torch.cat((feature_buffer_label, feature_buffer_unlabel))
    
        if encoder_name == "MLP":
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(X_combined)[-1].view(1, -1)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            #graph = generate_kNN_Setting1(X_combined, nearest_k, direction=direction)
            graph = generate_kNN_Setting1_ver2(X_combined, nearest_k, direction=direction)
            
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(graph, X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(graph, X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0:
            print(f"{len(feature_buffer_label)} labeled, {len(feature_buffer_unlabel)} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
            if encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
                print(graph)
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        if encoder_name == "MLP":
            loss = criterion(model(feature)[:len(feature_buffer_label)], labels)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            loss = criterion(model(graph, X_combined)[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies


def SSL_online_kNN_online_update_with_memory_online_kmeans_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = OnlineKMeansCosine(args.memory, device=device)

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "MLP":
        model = SimpleMLP(dataset_name).to(device)
    elif encoder_name == "GCN":
        model = SimpleGCN(dataset_name).to(device)
    elif encoder_name == "GraphSage":
        model = SimpleGraphSage(dataset_name).to(device)
    elif encoder_name == "GAT":
        model = SimpleGAT(dataset_name).to(device)
    elif encoder_name == "TAG":
        model = SimpleTAG(dataset_name).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
            X_combined = torch.cat((feature_buffer_label, memory.centroids))
        else:
            #feature_buffer_unlabel = torch.cat((feature_buffer_unlabel, input))
            X_combined = torch.cat((feature_buffer_label, memory.centroids, input))
            memory.partial_fit(input)

        if encoder_name == "MLP":
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(X_combined)[-1].view(1, -1)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            graph = generate_kNN_Setting1(X_combined, nearest_k, direction=direction)
            
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(graph, X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(graph, X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.centroids)} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
            if encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
                print(graph)
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        if encoder_name == "MLP":
            loss = criterion(model(feature)[:len(feature_buffer_label)], labels)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            loss = criterion(model(graph, X_combined)[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_online_kmeans_multi_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = OnlineKMeansCosine_Multi(args.memory, nearest_k, device=device)

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "MLP":
        model = SimpleMLP(dataset_name).to(device)
    elif encoder_name == "GCN":
        model = SimpleGCN(dataset_name).to(device)
    elif encoder_name == "GraphSage":
        model = SimpleGraphSage(dataset_name).to(device)
    elif encoder_name == "GAT":
        model = SimpleGAT(dataset_name).to(device)
    elif encoder_name == "TAG":
        model = SimpleTAG(dataset_name).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
            X_combined = torch.cat((feature_buffer_label, memory.centroids))
        else:
            #feature_buffer_unlabel = torch.cat((feature_buffer_unlabel, input))
            X_combined = torch.cat((feature_buffer_label, memory.centroids, input))
            memory.partial_fit(input)

        if encoder_name == "MLP":
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(X_combined)[-1].view(1, -1)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            graph = generate_kNN_Setting1(X_combined, nearest_k, direction=direction)
            
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(graph, X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(graph, X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.centroids)} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
            if encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
                print(graph)
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        if encoder_name == "MLP":
            loss = criterion(model(feature)[:len(feature_buffer_label)], labels)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            loss = criterion(model(graph, X_combined)[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_StreamKMpp_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = StreamKMpp(k=args.memory, buffer_size=args.memory, coreset_size=args.memory, device=device)

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "MLP":
        model = SimpleMLP(dataset_name).to(device)
    elif encoder_name == "GCN":
        model = SimpleGCN(dataset_name).to(device)
    elif encoder_name == "GraphSage":
        model = SimpleGraphSage(dataset_name).to(device)
    elif encoder_name == "GAT":
        model = SimpleGAT(dataset_name).to(device)
    elif encoder_name == "TAG":
        model = SimpleTAG(dataset_name).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = feature_buffer_label
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory))
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else:
            #feature_buffer_unlabel = torch.cat((feature_buffer_unlabel, input))
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = torch.cat((feature_buffer_label, input))
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    #print(feature_buffer_label.shape, buffered_memory.shape, input.shape)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory, input))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
            memory.partial_fit(input.squeeze(0))

        if encoder_name == "MLP":
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(X_combined)[-1].view(1, -1)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            graph = generate_kNN_Setting1(X_combined, nearest_k, direction=direction)
            
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(graph, X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(graph, X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
            if encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
                print(graph)
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        if encoder_name == "MLP":
            loss = criterion(model(feature)[:len(feature_buffer_label)], labels)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            loss = criterion(model(graph, X_combined)[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_StreamKMpp_Cosine_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    #memory = StreamKMpp_Cosine(k=args.memory, buffer_size=args.memory, coreset_size=args.memory, device=device)
    memory = StreamKMpp_Cosine(buffer_size=args.memory, device=device)

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "MLP":
        model = SimpleMLP(dataset_name).to(device)
    elif encoder_name == "GCN":
        model = SimpleGCN(dataset_name).to(device)
    elif encoder_name == "GraphSage":
        model = SimpleGraphSage(dataset_name).to(device)
    elif encoder_name == "GAT":
        model = SimpleGAT(dataset_name).to(device)
    elif encoder_name == "TAG":
        model = SimpleTAG(dataset_name).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = feature_buffer_label
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory))
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else:
            #feature_buffer_unlabel = torch.cat((feature_buffer_unlabel, input))
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = torch.cat((feature_buffer_label, input))
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    #print(feature_buffer_label.shape, buffered_memory.shape, input.shape)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory, input))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
            memory.partial_fit(input.squeeze(0))

        if encoder_name == "MLP":
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(X_combined)[-1].view(1, -1)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            graph = generate_kNN_Setting1(X_combined, nearest_k, direction=direction)
            
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(graph, X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(graph, X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
            if encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
                print(graph)
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        if encoder_name == "MLP":
            loss = criterion(model(feature)[:len(feature_buffer_label)], labels)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            loss = criterion(model(graph, X_combined)[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_StreamKMpp_Cosine_ver2_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    #memory = StreamKMpp_Cosine(k=args.memory, buffer_size=args.memory, coreset_size=args.memory, device=device)
    memory = StreamKMpp_Cosine(buffer_size=args.memory, device=device)

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "MLP":
        model = SimpleMLP(dataset_name).to(device)
    elif encoder_name == "GCN":
        model = SimpleGCN_edge_weight(dataset_name, args.gcn_layer).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = feature_buffer_label
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory))
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else:
            #feature_buffer_unlabel = torch.cat((feature_buffer_unlabel, input))
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = torch.cat((feature_buffer_label, input))
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    #print(feature_buffer_label.shape, buffered_memory.shape, input.shape)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory, input))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
            memory.partial_fit(input.squeeze(0))

        if encoder_name == "MLP":
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(X_combined)[-1].view(1, -1)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            graph = generate_kNN_Setting1_ver2(X_combined, nearest_k, direction=direction)
            
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(graph, X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(graph, X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
            if encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
                print(graph)
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        if encoder_name == "MLP":
            loss = criterion(model(feature)[:len(feature_buffer_label)], labels)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            loss = criterion(model(graph, X_combined)[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_StreamKMpp_Cosine_Time_ver2_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    #memory = StreamKMpp_Cosine(buffer_size=args.memory, device=device)
    memory = StreamKMpp_Cosine_Time(buffer_size=args.memory, device=device, time_threshold=args.time_threshold)

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "MLP":
        model = SimpleMLP(dataset_name).to(device)
    elif encoder_name == "GCN":
        model = SimpleGCN_edge_weight(dataset_name).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
            temp = memory.get_centroids()
            if temp is None:
                X_combined = feature_buffer_label if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer)))
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else:
            temp = memory.get_centroids()
            if temp is None:
                X_combined = torch.cat((feature_buffer_label, input)) if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer), input))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
            memory.partial_fit(input.squeeze(0), idx)

        if encoder_name == "MLP":
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1) if isItLabel else model(X_combined)[-1].view(1, -1)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            graph = generate_kNN_Setting1_ver2(X_combined, nearest_k, direction=direction)
            
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                outputs = model(graph, X_combined)[:len(feature_buffer_label)][-1].view(1, -1) if isItLabel else model(graph, X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
            if encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
                print(graph)
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        if encoder_name == "MLP":
            loss = criterion(model(feature)[:len(feature_buffer_label)], labels)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            loss = criterion(model(graph, X_combined)[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_StreamKMpp_Cosine_Soft_ver2_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = StreamKMpp_Cosine(buffer_size=args.memory, device=device)
    lamb_soft = args.lamb_soft

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "MLP":
        model = SimpleMLP(dataset_name).to(device)
    elif encoder_name == "GCN":
        model = SimpleGCN_edge_weight(dataset_name).to(device)
    elif encoder_name == "GraphSage":
        model = SimpleGraphSage_edge_weight(dataset_name).to(device)
    elif encoder_name == "GAT":
        model = SimpleGAT_edge_weight(dataset_name).to(device)
    elif encoder_name == "TAG":
        model = SimpleTAG_edge_weight(dataset_name).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, labels = torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
            temp = memory.get_centroids()
            if temp is None:
                X_combined = feature_buffer_label if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer)))
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else:
            temp = memory.get_centroids()
            if temp is None:
                X_combined = torch.cat((feature_buffer_label, input)) if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer), input))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
            memory.partial_fit(input.squeeze(0))

        if encoder_name == "MLP":
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1) if isItLabel else model(X_combined)[-1].view(1, -1)
                
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            graph = generate_kNN_Setting1_ver2(X_combined, nearest_k, direction=direction)
            
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                outputs = model(graph, X_combined)[:len(feature_buffer_label)][-1].view(1, -1) if isItLabel else model(graph, X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
            if encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
                print(graph)
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        #if encoder_name == "MLP":
        #    loss = criterion(model(feature)[:len(feature_buffer_label)], labels)
        #elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
        #    loss = criterion(model(graph, X_combined)[:len(feature_buffer_label)], labels)
        outputs = model(graph, X_combined)
        loss = criterion(outputs[:len(feature_buffer_label)], labels)
        preds = torch.argmax(outputs, dim=1)
        loss += lamb_soft * criterion(outputs[len(feature_buffer_label):], preds[len(feature_buffer_label):])
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_StreamKMpp_Cosine_Soft_Label_Time_ver2_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = StreamKMpp_Cosine(buffer_size=args.memory, device=device)
    lamb_soft = args.lamb_soft
    label_time_budget = args.label_time_budget

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "MLP":
        model = SimpleMLP(dataset_name).to(device)
    elif encoder_name == "GCN":
        model = SimpleGCN_edge_weight(dataset_name).to(device)
    elif encoder_name == "GraphSage":
        model = SimpleGraphSage_edge_weight(dataset_name).to(device)
    elif encoder_name == "GAT":
        model = SimpleGAT_edge_weight(dataset_name).to(device)
    elif encoder_name == "TAG":
        model = SimpleTAG_edge_weight(dataset_name).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    
                
    test_accuracies = []
    feature_buffer_label, labels = torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    time_buffer_label = torch.empty(0, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
            time_buffer_label = torch.cat((time_buffer_label, torch.tensor([idx], device=device)))
            
            minimum_time = idx - label_time_budget
            mask = time_buffer_label > minimum_time

            feature_buffer_label = feature_buffer_label[mask]
            labels = labels[mask]
            time_buffer_label = time_buffer_label[mask]
            
            temp = memory.get_centroids()
            if temp is None:
                X_combined = feature_buffer_label if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer)))
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else:
            temp = memory.get_centroids()
            if temp is None:
                X_combined = torch.cat((feature_buffer_label, input)) if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer), input))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
            memory.partial_fit(input.squeeze(0))

        if encoder_name == "MLP":
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1) if isItLabel else model(X_combined)[-1].view(1, -1)
                
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            graph = generate_kNN_Setting1_ver2(X_combined, nearest_k, direction=direction)
            
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                outputs = model(graph, X_combined)[:len(feature_buffer_label)][-1].view(1, -1) if isItLabel else model(graph, X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
            if encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
                print(graph)
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        #if encoder_name == "MLP":
        #    loss = criterion(model(feature)[:len(feature_buffer_label)], labels)
        #elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
        #    loss = criterion(model(graph, X_combined)[:len(feature_buffer_label)], labels)
        outputs = model(graph, X_combined)
        loss = criterion(outputs[:len(feature_buffer_label)], labels)
        preds = torch.argmax(outputs, dim=1)
        loss += lamb_soft * criterion(outputs[len(feature_buffer_label):], preds[len(feature_buffer_label):])
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_StreamKMpp_Cosine_Label_Time_ver2_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = StreamKMpp_Cosine(buffer_size=args.memory, device=device)
    label_time_budget = args.label_time_budget

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "MLP":
        model = SimpleMLP(dataset_name).to(device)
    elif encoder_name == "GCN":
        model = SimpleGCN_edge_weight(dataset_name, args.gcn_layer).to(device)
    elif encoder_name == "GraphSage":
        model = SimpleGraphSage_edge_weight(dataset_name).to(device)
    elif encoder_name == "GAT":
        model = SimpleGAT_edge_weight(dataset_name).to(device)
    elif encoder_name == "TAG":
        model = SimpleTAG_edge_weight(dataset_name).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, labels = torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    time_buffer_label = torch.empty(0, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
            time_buffer_label = torch.cat((time_buffer_label, torch.tensor([idx], device=device)))
            
            minimum_time = idx - label_time_budget
            mask = time_buffer_label > minimum_time

            feature_buffer_label = feature_buffer_label[mask]
            labels = labels[mask]
            time_buffer_label = time_buffer_label[mask]
            
            temp = memory.get_centroids()
            if temp is None:
                X_combined = feature_buffer_label if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer)))
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else:
            temp = memory.get_centroids()
            if temp is None:
                X_combined = torch.cat((feature_buffer_label, input)) if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer), input))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
            memory.partial_fit(input.squeeze(0))

        if encoder_name == "MLP":
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1) if isItLabel else model(X_combined)[-1].view(1, -1)
                
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            graph = generate_kNN_Setting1_ver2(X_combined, nearest_k, direction=direction)
            
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                outputs = model(graph, X_combined)[:len(feature_buffer_label)][-1].view(1, -1) if isItLabel else model(graph, X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
            if encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
                print(graph)
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        if encoder_name == "MLP":
            loss = criterion(model(feature)[:len(feature_buffer_label)], labels)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            loss = criterion(model(graph, X_combined)[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_StreamKMpp_Cosine_GSL_reg_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    #memory = StreamKMpp_Cosine(k=args.memory, buffer_size=args.memory, coreset_size=args.memory, device=device)
    memory = StreamKMpp_Cosine(buffer_size=args.memory, device=device)
    lamb = args.lamb

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "GCN":
        model = Simple_GSL(args).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = feature_buffer_label
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory))
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else:
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = torch.cat((feature_buffer_label, input))
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory, input))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
            memory.partial_fit(input.squeeze(0))
        # --- Test Phase ---
        model.eval()
        with torch.no_grad():
            if isItLabel:
                outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
            else:
                outputs = model(X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_combined)[:len(feature_buffer_label)], labels)
        loss += lamb * model.reg(X_combined)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_APPNP_StreamKMpp_Cosine_GSL_reg_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    #memory = StreamKMpp_Cosine(k=args.memory, buffer_size=args.memory, coreset_size=args.memory, device=device)
    memory = StreamKMpp_Cosine(buffer_size=args.memory, device=device)
    lamb = args.lamb

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "GCN":
        model = Simple_GSL_APPNP_correct(args).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = feature_buffer_label
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory))
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else:
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = torch.cat((feature_buffer_label, input))
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory, input))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
            memory.partial_fit(input.squeeze(0))
        # --- Test Phase ---
        model.eval()
        with torch.no_grad():
            if isItLabel:
                outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
            else:
                outputs = model(X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_combined)[:len(feature_buffer_label)], labels)
        loss += lamb * model.reg(X_combined)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_StreamKMpp_Cosine_GSL_reg_ver2_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = StreamKMpp_Cosine(buffer_size=args.memory, device=device)
    lamb = args.lamb

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "GCN":
        model = Simple_GSL(args).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = feature_buffer_label
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory))
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else:
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = torch.cat((feature_buffer_label, input))
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory, input))
                memory.partial_fit(input.squeeze(0))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
                model.eval()
                memory_emb = model.edge_scorer.internal_forward(temp)
                input_emb = model.edge_scorer.internal_forward(input)
                memory.partial_fit_ver2(input, input_emb, memory_emb)
            
        # --- Test Phase ---
        model.eval()
        with torch.no_grad():
            if isItLabel:
                outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
            else:
                outputs = model(X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_combined)[:len(feature_buffer_label)], labels)
        loss += lamb * model.reg(X_combined)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_StreamKMpp_Cosine_GSL_reg_ver2_bn_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = StreamKMpp_Cosine(buffer_size=args.memory, device=device)
    lamb = args.lamb

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "GCN":
        model = Simple_GSL_bn(args).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = feature_buffer_label
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory))
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else:
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = torch.cat((feature_buffer_label, input))
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory, input))
                memory.partial_fit(input.squeeze(0))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
                model.eval()
                memory_emb = model.edge_scorer.internal_forward(temp)
                input_emb = model.edge_scorer.internal_forward(input)
                memory.partial_fit_ver2(input, input_emb, memory_emb)
            
        # --- Test Phase ---
        model.eval()
        with torch.no_grad():
            if isItLabel:
                outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
            else:
                outputs = model(X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_combined)[:len(feature_buffer_label)], labels)
        loss += lamb * model.reg(X_combined)
        loss.backward()
        optimizer.step()
    return test_accuracies

def online_SSL_kNN_memory_Setting1(loader, dataset_size, args, lr, wd):
    opt_name, dataset_name, encoder_name, label_ratio, nearest_k, direction, memory_constant = args.opt, args.dataset, args.encoder, args.label_ratio, args.k, args.directed, args.memory_constant
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    time_decay_tau = args.time_decay_tau
    
    unlabeled_memory_size = args.memory
    if memory_constant:
        labeled_memory_size = args.labeled_size
        labeled_memory_type = args.labeled_memory_type 
        unlabeled_memory_size -= labeled_memory_size
    else: # full
        labeled_memory_size = 0
        
    if args.memory_type == "StreamKMpp_Cosine":
        memory = StreamKMpp_Cosine(buffer_size=unlabeled_memory_size, device=device)
    elif args.memory_type == "Window":
        memory = Window(buffer_size=unlabeled_memory_size, device=device, isLabel=False)
    elif args.memory_type == "Update":
        memory = Update(buffer_size=unlabeled_memory_size, device=device)
        
    if labeled_memory_type == "Window":
        labeled_memory = Window(buffer_size=labeled_memory_size, device=device)
    elif labeled_memory_type == "Window_time_decaying":
        labeled_memory = Window_time_decaying(buffer_size=labeled_memory_size, device=device)

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]
    LABELS[true_indices] = True
    print(LABELS)

    #criterion = nn.CrossEntropyLoss()
    if "time_decaying" in labeled_memory_type:
        criterion = nn.CrossEntropyLoss(reduction='none')
        criterion2 = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "GCN":
        model = SimpleGCN_edge_weight(dataset_name, args.gcn_layer).to(device)
    print(model)
        
    if opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    #feature_buffer_label, labels = torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel: # Labeled memory update
            labeled_memory.partial_fit(feat=input.squeeze(0), label=label.squeeze(0))

            feature_buffer_label, labels = labeled_memory.get_centroids()
            temp = memory.get_centroids()
            if temp is None:
                if "Update" in args.memory_type:
                    X_combined = feature_buffer_label
                else:
                    X_combined = feature_buffer_label if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer))) 
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else: # Unlabeled memory update
            feature_buffer_label, labels = labeled_memory.get_centroids()
            temp = memory.get_centroids()
            if temp is None:
                if "Update" in args.memory_type:
                     X_combined = torch.cat((feature_buffer_label, input))
                else:
                    X_combined = torch.cat((feature_buffer_label, input)) if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer), input))
                memory.partial_fit(input.squeeze(0))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
                memory.partial_fit(input.squeeze(0))
        
        # --- Test Phase ---
        model.eval()
        with torch.no_grad():
            graph = generate_kNN_Setting1_ver2(X_combined, nearest_k, direction=direction)
            outputs = model(graph, X_combined)[:len(feature_buffer_label)][-1].view(1, -1) if isItLabel else model(graph, X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        #loss = criterion(model(graph, X_combined)[:len(feature_buffer_label)], labels)
        preds = model(graph, X_combined)
        if "time_decaying" in labeled_memory_type:
            loss_temp = criterion(preds[:len(feature_buffer_label)], labels)
            loss = torch.mean(loss_temp * torch.exp(labeled_memory.get_time_decaying()/time_decay_tau))
        else:
            loss = criterion(preds[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def online_SSL_kNN_memory_ver2_Setting1(loader, dataset_size, args, lr, wd):
    opt_name, dataset_name, encoder_name, label_ratio, nearest_k, direction, memory_constant = args.opt, args.dataset, args.encoder, args.label_ratio, args.k, args.directed, args.memory_constant
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    time_decay_tau = args.time_decay_tau
    
    unlabeled_memory_size = args.memory
    if memory_constant:
        labeled_memory_size = args.labeled_size
        labeled_memory_type = args.labeled_memory_type 
        unlabeled_memory_size -= labeled_memory_size
    else: # full
        labeled_memory_size = 0
        
    if args.memory_type == "StreamKMpp_Cosine":
        memory = StreamKMpp_Cosine(buffer_size=unlabeled_memory_size, device=device)
    elif args.memory_type == "Window":
        memory = Window(buffer_size=unlabeled_memory_size, device=device, isLabel=False)
    elif args.memory_type == "Update":
        memory = Update(buffer_size=unlabeled_memory_size, device=device)
        
    if labeled_memory_type == "Window":
        labeled_memory = Window(buffer_size=labeled_memory_size, device=device)
    elif labeled_memory_type == "Window_time_decaying":
        labeled_memory = Window_time_decaying(buffer_size=labeled_memory_size, device=device)

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]
    LABELS[true_indices] = True
    print(LABELS)

    #criterion = nn.CrossEntropyLoss()
    if "time_decaying" in labeled_memory_type:
        criterion = nn.CrossEntropyLoss(reduction='none')
        criterion2 = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "GCN":
        model = SimpleGCN_edge_weight(dataset_name, args.gcn_layer).to(device)
    print(model)
        
    if opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    #feature_buffer_label, labels = torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel: # Labeled memory update
            labeled_memory.partial_fit(feat=input.squeeze(0), label=label.squeeze(0))

            feature_buffer_label, labels = labeled_memory.get_centroids()
            temp = memory.get_centroids()
            if temp is None:
                if "Update" in args.memory_type:
                    X_combined = feature_buffer_label
                else:
                    X_combined = feature_buffer_label if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer))) 
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else: # Unlabeled memory update
            feature_buffer_label, labels = labeled_memory.get_centroids()
            temp = memory.get_centroids()
            if temp is None:
                if "Update" in args.memory_type:
                     X_combined = torch.cat((feature_buffer_label, input))
                else:
                    X_combined = torch.cat((feature_buffer_label, input)) if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer), input))
                memory.partial_fit(input.squeeze(0))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
                #memory.partial_fit(input.squeeze(0))

                model.eval()
                graph = generate_kNN_Setting1_ver2(X_combined, nearest_k, direction=direction)
                outputs = model.get_embedding(graph, X_combined)
                
                memory_emb = outputs[len(feature_buffer_label):-1]
                if memory_emb.shape[0] != temp.shape[0]:
                    print("Error", memory_emb.shape, temp.shape)
                input_emb = outputs[-1].unsqueeze(0)
                memory.partial_fit_ver2(input, input_emb, memory_emb)
        
        # --- Test Phase ---
        model.eval()
        with torch.no_grad():
            graph = generate_kNN_Setting1_ver2(X_combined, nearest_k, direction=direction)
            outputs = model(graph, X_combined)[:len(feature_buffer_label)][-1].view(1, -1) if isItLabel else model(graph, X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        #loss = criterion(model(graph, X_combined)[:len(feature_buffer_label)], labels)
        preds = model(graph, X_combined)
        if "time_decaying" in labeled_memory_type:
            loss_temp = criterion(preds[:len(feature_buffer_label)], labels)
            loss = torch.mean(loss_temp * torch.exp(labeled_memory.get_time_decaying()/time_decay_tau))
        else:
            loss = criterion(preds[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def online_SSL_MLP_no_memory_Setting1(loader, dataset_size, args, lr, wd):
    opt_name, dataset_name, encoder_name, label_ratio, direction = args.opt, args.dataset, args.encoder, args.label_ratio, args.directed
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]
    LABELS[true_indices] = True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    #if encoder_name == "GCN":
    #    model = SimpleGCN_edge_weight(dataset_name, args.gcn_layer).to(device)
    if encoder_name == "MLP":
        model = SimpleMLP_bn(dataset_name, args.gcn_layer).to(device)
        
    print(model)
        
    if opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]
        
        # --- Test Phase ---
        model.eval()
        with torch.no_grad():
            outputs = model(input).view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        if isItLabel:
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(input), label)
            loss.backward()
            optimizer.step()
    return test_accuracies

def online_SSL_MLP_memory_Setting1(loader, dataset_size, args, lr, wd):
    opt_name, dataset_name, encoder_name, label_ratio, direction, memory_constant = args.opt, args.dataset, args.encoder, args.label_ratio, args.directed, args.memory_constant
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    time_decay_tau = args.time_decay_tau
    
    unlabeled_memory_size = args.memory
    if memory_constant:
        labeled_memory_size = args.labeled_size
        labeled_memory_type = args.labeled_memory_type 
        unlabeled_memory_size -= labeled_memory_size
    else: # full
        labeled_memory_size = 0
        
    if args.memory_type == "StreamKMpp_Cosine":
        memory = StreamKMpp_Cosine(buffer_size=unlabeled_memory_size, device=device)
    elif args.memory_type == "Window":
        memory = Window(buffer_size=unlabeled_memory_size, device=device, isLabel=False)
    elif args.memory_type == "Update":
        memory = Update(buffer_size=unlabeled_memory_size, device=device)
        
    if labeled_memory_type == "Window":
        labeled_memory = Window(buffer_size=labeled_memory_size, device=device)
    elif labeled_memory_type == "Window_time_decaying":
        labeled_memory = Window_time_decaying(buffer_size=labeled_memory_size, device=device)

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]
    LABELS[true_indices] = True
    print(LABELS)

    #criterion = nn.CrossEntropyLoss()
    if "time_decaying" in labeled_memory_type:
        criterion = nn.CrossEntropyLoss(reduction='none')
        criterion2 = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    #if encoder_name == "GCN":
    #    model = SimpleGCN_edge_weight(dataset_name, args.gcn_layer).to(device)
    if encoder_name == "MLP":
        model = SimpleMLP_bn(dataset_name, args.gcn_layer).to(device)
        
    print(model)
        
    if opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel: # Labeled memory update
            labeled_memory.partial_fit(feat=input.squeeze(0), label=label.squeeze(0))

            feature_buffer_label, labels = labeled_memory.get_centroids()
            temp = memory.get_centroids()
            if temp is None:
                if "Update" in args.memory_type:
                    X_combined = feature_buffer_label
                else:
                    X_combined = feature_buffer_label if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer))) 
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else: # Unlabeled memory update
            feature_buffer_label, labels = labeled_memory.get_centroids()
            temp = memory.get_centroids()
            if temp is None:
                if "Update" in args.memory_type:
                     X_combined = torch.cat((feature_buffer_label, input))
                else:
                    X_combined = torch.cat((feature_buffer_label, input)) if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer), input))
                memory.partial_fit(input.squeeze(0))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
                memory.partial_fit(input.squeeze(0))
        
        # --- Test Phase ---
        model.eval()
        with torch.no_grad():
            outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1) if isItLabel else model(X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        #loss = criterion(model(graph, X_combined)[:len(feature_buffer_label)], labels)
        preds = model(X_combined)
        if "time_decaying" in labeled_memory_type:
            loss_temp = criterion(preds[:len(feature_buffer_label)], labels)
            loss = torch.mean(loss_temp * torch.exp(labeled_memory.get_time_decaying()/time_decay_tau))
        else:
            loss = criterion(preds[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def online_SSL_MLP_memory_ver2_Setting1(loader, dataset_size, args, lr, wd):
    opt_name, dataset_name, encoder_name, label_ratio, direction, memory_constant = args.opt, args.dataset, args.encoder, args.label_ratio, args.directed, args.memory_constant
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    time_decay_tau = args.time_decay_tau
    
    unlabeled_memory_size = args.memory
    if memory_constant:
        labeled_memory_size = args.labeled_size
        labeled_memory_type = args.labeled_memory_type 
        unlabeled_memory_size -= labeled_memory_size
    else: # full
        labeled_memory_size = 0
        
    if args.memory_type == "StreamKMpp_Cosine":
        memory = StreamKMpp_Cosine(buffer_size=unlabeled_memory_size, device=device)
    elif args.memory_type == "Window":
        memory = Window(buffer_size=unlabeled_memory_size, device=device, isLabel=False)
    elif args.memory_type == "Update":
        memory = Update(buffer_size=unlabeled_memory_size, device=device)
        
    if labeled_memory_type == "Window":
        labeled_memory = Window(buffer_size=labeled_memory_size, device=device)
    elif labeled_memory_type == "Window_time_decaying":
        labeled_memory = Window_time_decaying(buffer_size=labeled_memory_size, device=device)

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]
    LABELS[true_indices] = True
    print(LABELS)

    #criterion = nn.CrossEntropyLoss()
    if "time_decaying" in labeled_memory_type:
        criterion = nn.CrossEntropyLoss(reduction='none')
        criterion2 = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    #if encoder_name == "GCN":
    #    model = SimpleGCN_edge_weight(dataset_name, args.gcn_layer).to(device)
    if encoder_name == "MLP":
        model = SimpleMLP_bn(dataset_name, args.gcn_layer).to(device)
        
    print(model)
        
    if opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel: # Labeled memory update
            labeled_memory.partial_fit(feat=input.squeeze(0), label=label.squeeze(0))

            feature_buffer_label, labels = labeled_memory.get_centroids()
            temp = memory.get_centroids()
            if temp is None:
                if "Update" in args.memory_type:
                    X_combined = feature_buffer_label
                else:
                    X_combined = feature_buffer_label if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer))) 
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else: # Unlabeled memory update
            feature_buffer_label, labels = labeled_memory.get_centroids()
            temp = memory.get_centroids()
            if temp is None:
                if "Update" in args.memory_type:
                     X_combined = torch.cat((feature_buffer_label, input))
                else:
                    X_combined = torch.cat((feature_buffer_label, input)) if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer), input))
                memory.partial_fit(input.squeeze(0))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
                #memory.partial_fit(input.squeeze(0))

                model.eval()
                outputs = model.get_embedding(X_combined)
                
                memory_emb = outputs[len(feature_buffer_label):-1]
                if memory_emb.shape[0] != temp.shape[0]:
                    print("Error", memory_emb.shape, temp.shape)
                input_emb = outputs[-1].unsqueeze(0)
                memory.partial_fit_ver2(input, input_emb, memory_emb)
        
        # --- Test Phase ---
        model.eval()
        with torch.no_grad():
            outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1) if isItLabel else model(X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        #loss = criterion(model(graph, X_combined)[:len(feature_buffer_label)], labels)
        preds = model(X_combined)
        if "time_decaying" in labeled_memory_type:
            loss_temp = criterion(preds[:len(feature_buffer_label)], labels)
            loss = torch.mean(loss_temp * torch.exp(labeled_memory.get_time_decaying()/time_decay_tau))
        else:
            loss = criterion(preds[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def online_SSL_kNN_memory_GSL_Setting1(loader, dataset_size, args, lr, wd):
    opt_name, dataset_name, encoder_name, label_ratio, nearest_k, direction, lamb, memory_constant = args.opt, args.dataset, args.encoder, args.label_ratio, args.k, args.directed, args.lamb, args.memory_constant
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    time_decay_tau = args.time_decay_tau
    
    unlabeled_memory_size = args.memory
    if memory_constant:
        labeled_memory_size = args.labeled_size
        labeled_memory_type = args.labeled_memory_type 
        unlabeled_memory_size -= labeled_memory_size
    else: # full
        labeled_memory_size = 0
        
    if args.memory_type == "StreamKMpp_Cosine":
        memory = StreamKMpp_Cosine(buffer_size=unlabeled_memory_size, device=device)
    elif args.memory_type == "Update":
        memory = Update(buffer_size=unlabeled_memory_size, device=device)
    elif args.memory_type == "Update_ver2":
        memory = Update_ver2(buffer_size=unlabeled_memory_size, device=device)
    elif args.memory_type == "Update_min":
        memory = Update_min(buffer_size=unlabeled_memory_size, device=device)
    elif args.memory_type == "Update_topk":
        memory = Update_topk(buffer_size=unlabeled_memory_size, device=device, k=nearest_k)
    elif args.memory_type == "Window":
        memory = Window(buffer_size=unlabeled_memory_size, device=device, isLabel=False)
        
    if labeled_memory_type == "Window":
        labeled_memory = Window(buffer_size=labeled_memory_size, device=device)
    elif labeled_memory_type == "Window_time_decaying":
        labeled_memory = Window_time_decaying(buffer_size=labeled_memory_size, device=device)
        
    GSL_type = args.GSL_type # GSL_reg_ver2_bn

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)
    if "init" in GSL_type:
        init = 1
        LABELS[:init] = True
        remaining_size = dataset_size - init
        num_true_remaining = int(label_ratio * remaining_size)
        true_indices = torch.randperm(remaining_size)[:num_true_remaining] + init
    else:
        true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]
    LABELS[true_indices] = True
    print(torch.sum(LABELS), LABELS)

    if "time_decaying" in labeled_memory_type:
        criterion = nn.CrossEntropyLoss(reduction='none')
        criterion2 = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "GCN":
        if "bn" in GSL_type:
            model = Simple_GSL_bn(args).to(device)
    print(model)
        
    if opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    #feature_buffer_label, labels = torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel: # Labeled memory update
            labeled_memory.partial_fit(feat=input.squeeze(0), label=label.squeeze(0))

            feature_buffer_label, labels = labeled_memory.get_centroids()
            temp = memory.get_centroids()
            if temp is None:
                if "Update" in args.memory_type:
                    X_combined = feature_buffer_label
                else:
                    X_combined = feature_buffer_label if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer))) 
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else: # Unlabeled memory update
            feature_buffer_label, labels = labeled_memory.get_centroids()
            temp = memory.get_centroids()
            if temp is None:
                if "Update" in args.memory_type:
                     X_combined = torch.cat((feature_buffer_label, input))
                else:
                    X_combined = torch.cat((feature_buffer_label, input)) if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer), input))
                memory.partial_fit(input.squeeze(0))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
                """
                if "ver2" in GSL_type and args.memory_type != "Window":
                    model.eval()
                    with torch.no_grad():
                        memory_emb = model.edge_scorer.internal_forward(temp)
                        input_emb = model.edge_scorer.internal_forward(input)
                    memory.partial_fit_ver2(input, input_emb, memory_emb)
                else:
                    memory.partial_fit(input.squeeze(0))
                """
                if args.memory_type != "Window":
                    if "ver1" in GSL_type: # feature-level update
                        memory.partial_fit(input.squeeze(0))
                    else:
                        if "ver2" in GSL_type: # score-level update
                            model.eval()
                            with torch.no_grad():
                                memory_emb = model.edge_scorer.internal_forward(temp)
                                input_emb = model.edge_scorer.internal_forward(input)
                        elif "ver3" in GSL_type: # emb-level update
                            model.eval()
                            with torch.no_grad():
                                memory_emb = model.get_embedding(temp)
                                input_emb = model.get_embedding(input)
                        memory.partial_fit_ver2(input, input_emb, memory_emb)
                else:
                    memory.partial_fit(input.squeeze(0))
        
        # --- Test Phase ---
        model.eval()
        
        with torch.no_grad():
            outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1) if isItLabel else model(X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        preds = model(X_combined)
        ########
        softmax = nn.Softmax(dim=1)
        probs = softmax(preds)
        pred_classes = torch.argmax(probs, dim=1)
        confidences = torch.max(probs, dim=1).values
        #######
        Label_confidence_avg = torch.mean(confidences[:len(feature_buffer_label)])
        confidence_mask = confidences[len(feature_buffer_label):] > Label_confidence_avg
        ########
        if "time_decaying" in labeled_memory_type:
            loss_temp = criterion(preds[:len(feature_buffer_label)], labels)
            loss = torch.mean(loss_temp * torch.exp(labeled_memory.get_time_decaying()/time_decay_tau))
        else:
            loss = criterion(preds[:len(feature_buffer_label)], labels)
        if "reg" in GSL_type:
            if "orthogonal" in GSL_type:
                loss += lamb * model.reg_orthogonal(X_combined)
            elif "_HER_" in GSL_type:
                loss += lamb * model.reg_HER(X_combined)
            elif "_HERD_" in GSL_type:
                loss += lamb * model.reg_HER_distance(X_combined)
            elif "_MI_" in GSL_type:
                loss += lamb * model.reg_MI(X_combined)
            elif "_eye_" in GSL_type:
                loss += lamb * model.reg_eye(X_combined)
            else:
                loss += lamb * model.reg(X_combined)
        if "conf" in GSL_type:
            if len(feature_buffer_label)>0:
                #print(preds[len(feature_buffer_label):][confidence_mask].shape, pred_classes[len(feature_buffer_label):][confidence_mask].shape)
                loss += args.lamb_conf * criterion2(preds[len(feature_buffer_label):][confidence_mask], pred_classes[len(feature_buffer_label):][confidence_mask])
        loss.backward()
        optimizer.step()
    return test_accuracies

import time
def online_SSL_kNN_memory_GSL_Setting1_time(loader, dataset_size, args, lr, wd):
    opt_name, dataset_name, encoder_name, label_ratio, nearest_k, direction, lamb, memory_constant = args.opt, args.dataset, args.encoder, args.label_ratio, args.k, args.directed, args.lamb, args.memory_constant
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    time_decay_tau = args.time_decay_tau
    
    unlabeled_memory_size = args.memory
    if memory_constant:
        labeled_memory_size = args.labeled_size
        labeled_memory_type = args.labeled_memory_type 
        unlabeled_memory_size -= labeled_memory_size
    else: # full
        labeled_memory_size = 0
        
    if args.memory_type == "StreamKMpp_Cosine":
        memory = StreamKMpp_Cosine(buffer_size=unlabeled_memory_size, device=device)
    elif args.memory_type == "Update":
        memory = Update(buffer_size=unlabeled_memory_size, device=device)
    elif args.memory_type == "Update_ver2":
        memory = Update_ver2(buffer_size=unlabeled_memory_size, device=device)
    elif args.memory_type == "Update_min":
        memory = Update_min(buffer_size=unlabeled_memory_size, device=device)
    elif args.memory_type == "Update_topk":
        memory = Update_topk(buffer_size=unlabeled_memory_size, device=device, k=nearest_k)
    elif args.memory_type == "Window":
        memory = Window(buffer_size=unlabeled_memory_size, device=device, isLabel=False)
        
    if labeled_memory_type == "Window":
        labeled_memory = Window(buffer_size=labeled_memory_size, device=device)
    elif labeled_memory_type == "Window_time_decaying":
        labeled_memory = Window_time_decaying(buffer_size=labeled_memory_size, device=device)
        
    GSL_type = args.GSL_type # GSL_reg_ver2_bn

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)
    if "init" in GSL_type:
        init = 1
        LABELS[:init] = True
        remaining_size = dataset_size - init
        num_true_remaining = int(label_ratio * remaining_size)
        true_indices = torch.randperm(remaining_size)[:num_true_remaining] + init
    else:
        true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]
    LABELS[true_indices] = True
    print(torch.sum(LABELS), LABELS)

    if "time_decaying" in labeled_memory_type:
        criterion = nn.CrossEntropyLoss(reduction='none')
        criterion2 = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "GCN":
        if "bn" in GSL_type:
            model = Simple_GSL_bn(args).to(device)
    print(model)
        
    if opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    memory_update_time, inference_time, training_time = [], [], []
    #feature_buffer_label, labels = torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        time_A = time.time()
        if isItLabel: # Labeled memory update
            labeled_memory.partial_fit(feat=input.squeeze(0), label=label.squeeze(0))

            feature_buffer_label, labels = labeled_memory.get_centroids()
            temp = memory.get_centroids()
            if temp is None:
                if "Update" in args.memory_type:
                    X_combined = feature_buffer_label
                else:
                    X_combined = feature_buffer_label if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer))) 
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else: # Unlabeled memory update
            feature_buffer_label, labels = labeled_memory.get_centroids()
            temp = memory.get_centroids()
            if temp is None:
                if "Update" in args.memory_type:
                     X_combined = torch.cat((feature_buffer_label, input))
                else:
                    X_combined = torch.cat((feature_buffer_label, input)) if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer), input))
                memory.partial_fit(input.squeeze(0))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
                if args.memory_type != "Window":
                    if "ver1" in GSL_type: # feature-level update
                        memory.partial_fit(input.squeeze(0))
                    else:
                        if "ver2" in GSL_type: # score-level update
                            model.eval()
                            with torch.no_grad():
                                memory_emb = model.edge_scorer.internal_forward(temp)
                                input_emb = model.edge_scorer.internal_forward(input)
                        elif "ver3" in GSL_type: # emb-level update
                            model.eval()
                            with torch.no_grad():
                                memory_emb = model.get_embedding(temp)
                                input_emb = model.get_embedding(input)
                        memory.partial_fit_ver2(input, input_emb, memory_emb)
                else:
                    memory.partial_fit(input.squeeze(0))
        time_B = time.time()
        # --- Test Phase ---
        model.eval()
        time_C = time.time()
        with torch.no_grad():
            outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1) if isItLabel else model(X_combined)[-1].view(1, -1)
        time_D = time.time()
        
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        time_E = time.time()
        preds = model(X_combined)
        ########
        if "time_decaying" in labeled_memory_type:
            loss_temp = criterion(preds[:len(feature_buffer_label)], labels)
            loss = torch.mean(loss_temp * torch.exp(labeled_memory.get_time_decaying()/time_decay_tau))
        else:
            loss = criterion(preds[:len(feature_buffer_label)], labels)
        if "reg" in GSL_type:
            if "orthogonal" in GSL_type:
                loss += lamb * model.reg_orthogonal(X_combined)
            elif "_HER_" in GSL_type:
                loss += lamb * model.reg_HER(X_combined)
            elif "_HERD_" in GSL_type:
                loss += lamb * model.reg_HER_distance(X_combined)
            elif "_MI_" in GSL_type:
                loss += lamb * model.reg_MI(X_combined)
            elif "_eye_" in GSL_type:
                loss += lamb * model.reg_eye(X_combined)
            else:
                loss += lamb * model.reg(X_combined)
        loss.backward()
        optimizer.step()
        time_F = time.time()
        memory_update_time.append(time_B-time_A)
        inference_time.append(time_D-time_C)
        training_time.append(time_F-time_E)
    return test_accuracies, memory_update_time, inference_time, training_time

def online_SSL_TLP_Setting1(loader, dataset_size, args):
    dataset_name, label_ratio, memory_constant = args.dataset, args.label_ratio, args.memory_constant
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    
    unlabeled_memory_size = args.memory
    if memory_constant:
        labeled_memory_size = args.labeled_size
        unlabeled_memory_size -= labeled_memory_size
    else: # full
        labeled_memory_size = 0

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]
    LABELS[true_indices] = True
    print(torch.sum(LABELS), LABELS)

    _, num_classes = dataset_adaptation(dataset_name)

    c = num_classes

    label_feats, label_labels, unlabel_feats = [], [], []
    label_index, unlabel_index = [], []
    
    unlabel_size = unlabeled_memory_size
    label_size = labeled_memory_size

    def rbf(a, b):
        # sigma 0.1, 1, 10 -> sigma^2 0.01 1 100 -> 1/sigma^2 100 1 0.01
        #rbfgamma = 0.01
        #rbfgamma = 1.0 
        rbfgamma = 100.0
        return torch.exp(-rbfgamma * torch.sum((a - b) ** 2, dim=1, keepdim=True))
    
    similarity = rbf
    C_C = torch.zeros((c, c), device=device)
    index = c - 1
                
    test_accuracies = []
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        isItLabel = LABELS[idx]

        feat = input
        C_C = C_C.to(device)
    
        # ===== your original logic, optimized for GPU =====
        prev_C = C_C.shape[0]
        C_C_new = torch.zeros((prev_C + 1, prev_C + 1), device=device)
        C_C_new[:prev_C, :prev_C] = C_C
        index += 1
    
        if isItLabel:
            lbl_tensor = torch.tensor([[label.item()]], dtype=torch.float32, device=device)
        else:
            lbl_tensor = None
    
        # ----- similarity with labeled -----
        if len(label_feats) > 0:
            label_feat_tensor = torch.cat(label_feats, dim=0)
            label_label_tensor = torch.cat(label_labels, dim=0)
    
            A = similarity(label_feat_tensor, feat)
            temp = []
            for cla in range(c):
                mask = (label_label_tensor.reshape(-1) == cla)
                temp.append(A[mask].mean().unsqueeze(0) if mask.any() else torch.zeros(1, device=device))
            r = torch.cat(temp).unsqueeze(0)
            C_C_new[:c, -1] = r.T.reshape(-1)
            C_C_new[-1, :c] = r.T.reshape(-1)
    
        # ----- similarity with all feats -----
        all_feats = label_feats + unlabel_feats
        if len(all_feats) > 0:
            feat_concat = torch.cat(all_feats, dim=0)
            temp = similarity(feat_concat, feat)
            C_C_new[c:-1, -1] = temp.reshape(-1)
            C_C_new[-1, c:-1] = temp.reshape(-1)
    
        # ----- update buffers -----
        if lbl_tensor is not None:
            label_feats.append(feat)
            label_labels.append(lbl_tensor)
            label_index.append(index)
        else:
            unlabel_feats.append(feat)
            unlabel_index.append(index)
    
        # ----- unlabel buffer full -----
        if len(unlabel_feats) > unlabel_size:
            target_index = unlabel_index.pop(0)
            unlabel_feats.pop(0)
            a = C_C_new[target_index].reshape(-1, 1)
            b = a @ a.T
            update = b / (torch.sum(a) + 1e-8)
            C_C_new += update
    
            mask = torch.ones(C_C_new.shape[0], dtype=torch.bool, device=device)
            mask[target_index] = False
            C_C_new = C_C_new[mask][:, mask]
    
            label_index = [x - 1 if x > target_index else x for x in label_index]
            unlabel_index = [x - 1 if x > target_index else x for x in unlabel_index]
            index -= 1
    
        # ----- label buffer full -----
        if len(label_feats) > label_size:
            target_index = label_index.pop(0)
            label_feats.pop(0)
            label_labels.pop(0)
            a = C_C_new[target_index].reshape(-1, 1)
            b = a @ a.T
            update = b / (torch.sum(a) + 1e-8)
            C_C_new += update
    
            mask = torch.ones(C_C_new.shape[0], dtype=torch.bool, device=device)
            mask[target_index] = False
            C_C_new = C_C_new[mask][:, mask]
    
            label_index = [x - 1 if x > target_index else x for x in label_index]
            unlabel_index = [x - 1 if x > target_index else x for x in unlabel_index]
            index -= 1
    
        # ----- graph propagation -----
        C_C = C_C_new
        A = (C_C + C_C.T) / 2
        D = torch.diag(A.sum(dim=1))
        L = D - A
        F_l = torch.eye(c, device=device)
        L_ul = L[c:, :c]
        L_uu = L[c:, c:]
    
        if L_uu.shape[0] > 0:
            #rhs = -L_ul @ F_l
            #F_u = torch.linalg.solve(L_uu + 1e-6 * torch.eye(L_uu.shape[0], device=device), rhs)
            rhs = -L_ul @ F_l
            I_reg = 1e-6 * torch.eye(L_uu.shape[0], device=device)
            F_u, *_ = torch.linalg.lstsq(L_uu + I_reg, rhs)
            if index - c < F_u.shape[0]:
                out = F_u[index - c]
                pred = torch.argmax(out).item()
    
                #if not isItLabel:  # evaluate on unlabeled samples
                #    correct += int(pred == label.item())
                accuracy = (pred == label.item())
                test_accuracies.append(accuracy)
    
        if (idx + 1) % 10000 == 0:
            print(f"[{idx+1}/{dataset_size}] Running accuracy: {np.mean(test_accuracies):.4f}  {len(unlabel_feats)} unlabeled, {len(label_feats)} labeled")
    return test_accuracies

def online_SSL_kNN_memory_GSL_ver2_Setting1(loader, dataset_size, args, lr, wd):
    opt_name, dataset_name, encoder_name, label_ratio, nearest_k, direction, lamb, memory_constant = args.opt, args.dataset, args.encoder, args.label_ratio, args.k, args.directed, args.lamb, args.memory_constant
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    
    unlabeled_memory_size = args.memory
    if memory_constant:
        labeled_memory_size = args.labeled_size
        labeled_memory_type = args.labeled_memory_type 
        unlabeled_memory_size -= labeled_memory_size
    else: # full
        labeled_memory_size = 0
        
    if args.memory_type == "StreamKMpp_Cosine":
        memory = StreamKMpp_Cosine(buffer_size=unlabeled_memory_size, device=device)
    elif args.memory_type == "Update":
        memory = Update(buffer_size=unlabeled_memory_size, device=device)
    elif args.memory_type == "Update_ver2":
        memory = Update_ver2(buffer_size=unlabeled_memory_size, device=device)
    elif args.memory_type == "Update_min":
        memory = Update_min(buffer_size=unlabeled_memory_size, device=device)
    elif args.memory_type == "Update_topk":
        memory = Update_topk(buffer_size=unlabeled_memory_size, device=device, k=nearest_k)
    elif args.memory_type == "Window":
        memory = Window(buffer_size=unlabeled_memory_size, device=device, isLabel=False)
        
    if labeled_memory_type == "Window":
        labeled_memory = Window(buffer_size=labeled_memory_size, device=device)
    elif labeled_memory_type == "Window_time_decaying":
        labeled_memory = Window_time_decaying(buffer_size=labeled_memory_size, device=device)
        
    GSL_type = args.GSL_type # GSL_reg_ver2_bn

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]
    LABELS[true_indices] = True
    print(torch.sum(LABELS), LABELS)

    if "time_decaying" in labeled_memory_type:
        criterion = nn.CrossEntropyLoss(reduction='none')
        criterion2 = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "GCN":
        if "bn" in GSL_type:
            model = Simple_GSL_bn(args).to(device)
    print(model)
        
    if opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    #feature_buffer_label, labels = torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        """
        if isItLabel: # Labeled memory update
            labeled_memory.partial_fit(feat=input.squeeze(0), label=label.squeeze(0))

            feature_buffer_label, labels = labeled_memory.get_centroids()
            temp = memory.get_centroids()
            if temp is None:
                if "Update" in args.memory_type:
                    X_combined = feature_buffer_label
                else:
                    X_combined = feature_buffer_label if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer))) 
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else: # Unlabeled memory update
            feature_buffer_label, labels = labeled_memory.get_centroids()
            temp = memory.get_centroids()
            if temp is None:
                if "Update" in args.memory_type:
                     X_combined = torch.cat((feature_buffer_label, input))
                else:
                    X_combined = torch.cat((feature_buffer_label, input)) if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer), input))
                memory.partial_fit(input.squeeze(0))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
                if "ver2" in GSL_type and args.memory_type != "Window":
                    model.eval()
                    with torch.no_grad():
                        memory_emb = model.edge_scorer.internal_forward(temp)
                        input_emb = model.edge_scorer.internal_forward(input)
                    memory.partial_fit_ver2(input, input_emb, memory_emb)
                else:
                    memory.partial_fit(input.squeeze(0))
        """
        if isItLabel: # Labeled memory update
            labeled_memory.partial_fit(feat=input.squeeze(0), label=label.squeeze(0))
 
        feature_buffer_label, labels = labeled_memory.get_centroids()
        temp = memory.get_centroids()
        if temp is None:
            if "Update" in args.memory_type:
                 X_combined = torch.cat((feature_buffer_label, input))
            else:
                X_combined = torch.cat((feature_buffer_label, input)) if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer), input))
            memory.partial_fit(input.squeeze(0))
        else:
            X_combined = torch.cat((feature_buffer_label, temp, input))
            if "ver2" in GSL_type and args.memory_type != "Window":
                model.eval()
                with torch.no_grad():
                    memory_emb = model.edge_scorer.internal_forward(temp)
                    input_emb = model.edge_scorer.internal_forward(input)
                memory.partial_fit_ver2(input, input_emb, memory_emb)
            else:
                memory.partial_fit(input.squeeze(0))
                    
        # --- Test Phase ---
        model.eval()
        with torch.no_grad():
            outputs = model(X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        preds = model(X_combined)
        ########
        softmax = nn.Softmax(dim=1)
        probs = softmax(preds)
        pred_classes = torch.argmax(probs, dim=1)
        confidences = torch.max(probs, dim=1).values
        #######
        Label_confidence_avg = torch.mean(confidences[:len(feature_buffer_label)])
        confidence_mask = confidences[len(feature_buffer_label):] > Label_confidence_avg
        ########
        if "time_decaying" in labeled_memory_type:
            loss_temp = criterion(preds[:len(feature_buffer_label)], labels)
            loss = torch.mean(loss_temp * torch.exp(labeled_memory.get_time_decaying()))
        else:
            loss = criterion(preds[:len(feature_buffer_label)], labels)
        if "reg" in GSL_type:
            if "orthogonal" in GSL_type:
                loss += lamb * model.reg_orthogonal(X_combined)
            elif "_HER_" in GSL_type:
                loss += lamb * model.reg_HER(X_combined)
            elif "_HERD_" in GSL_type:
                loss += lamb * model.reg_HER_distance(X_combined)
            elif "_MI_" in GSL_type:
                loss += lamb * model.reg_MI(X_combined)
            else:
                loss += lamb * model.reg(X_combined)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_StreamKMpp_Cosine_GSL_reg_ver2_bn_exp_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = StreamKMpp_Cosine(buffer_size=args.memory, device=device)
    lamb = args.lamb

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    if encoder_name == "GCN":
        model = Simple_GSL_bn(args).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    time_buffer_label = torch.empty(0, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
            time_buffer_label -= 1
            time_buffer_label = torch.cat((time_buffer_label, torch.tensor([0], device=device)))
            
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = feature_buffer_label
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory))
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else:
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = torch.cat((feature_buffer_label, input))
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory, input))
                memory.partial_fit(input.squeeze(0))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
                model.eval()
                memory_emb = model.edge_scorer.internal_forward(temp)
                input_emb = model.edge_scorer.internal_forward(input)
                memory.partial_fit_ver2(input, input_emb, memory_emb)
            
        # --- Test Phase ---
        model.eval()
        with torch.no_grad():
            if isItLabel:
                outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
            else:
                outputs = model(X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        #loss = criterion(model(X_combined)[:len(feature_buffer_label)], labels)
        loss_temp = criterion(model(X_combined)[:len(feature_buffer_label)], labels)
        loss = torch.mean(loss_temp * torch.exp(time_buffer_label))
        loss += lamb * model.reg(X_combined)
        loss.backward()
        optimizer.step()
    return test_accuracies    

def SSL_online_kNN_online_update_with_memory_StreamKMpp_Cosine_GSL_reg_ver2_bn_Label_Time_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = StreamKMpp_Cosine(buffer_size=args.memory, device=device)
    lamb = args.lamb
    label_time_budget = args.label_time_budget

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "GCN":
        model = Simple_GSL_bn(args).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    time_buffer_label = torch.empty(0, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
            time_buffer_label = torch.cat((time_buffer_label, torch.tensor([idx], device=device)))

            minimum_time = idx - label_time_budget
            mask = time_buffer_label > minimum_time

            feature_buffer_label = feature_buffer_label[mask]
            labels = labels[mask]
            time_buffer_label = time_buffer_label[mask]
            
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = feature_buffer_label
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory))
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else:
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = torch.cat((feature_buffer_label, input))
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory, input))
                memory.partial_fit(input.squeeze(0))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
                model.eval()
                memory_emb = model.edge_scorer.internal_forward(temp)
                input_emb = model.edge_scorer.internal_forward(input)
                memory.partial_fit_ver2(input, input_emb, memory_emb)
            
        # --- Test Phase ---
        model.eval()
        with torch.no_grad():
            if isItLabel:
                outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
            else:
                outputs = model(X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_combined)[:len(feature_buffer_label)], labels)
        loss += lamb * model.reg(X_combined)
        loss.backward()
        optimizer.step()
    return test_accuracies


def SSL_online_kNN_online_update_with_memory_StreamKMpp_Cosine_GSL_reg_ver2_bn_ver2_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = StreamKMpp_Cosine(buffer_size=args.memory, device=device)
    lamb = args.lamb

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "GCN":
        model = Simple_GSL_bn_ver2(args).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = feature_buffer_label
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory))
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else:
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = torch.cat((feature_buffer_label, input))
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory, input))
                memory.partial_fit(input.squeeze(0))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
                model.eval()
                memory_emb = model.edge_scorer.internal_forward(temp)
                input_emb = model.edge_scorer.internal_forward(input)
                memory.partial_fit_ver2(input, input_emb, memory_emb)
            
        # --- Test Phase ---
        model.eval()
        with torch.no_grad():
            if isItLabel:
                outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
            else:
                outputs = model(X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_combined)[:len(feature_buffer_label)], labels)
        loss += lamb * model.reg(X_combined)
        loss.backward()
        optimizer.step()
    return test_accuracies


def SSL_online_kNN_online_update_with_memory_StreamKMpp_Cosine_GSL_add_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = StreamKMpp_Cosine(k=args.memory, buffer_size=args.memory, coreset_size=args.memory, device=device)

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "GCN":
        model = Simple_GSL_add(args).to(device) # Simple_GSL(args).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = feature_buffer_label
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory))
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else:
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = torch.cat((feature_buffer_label, input))
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory, input))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
            memory.partial_fit(input.squeeze(0))
        # --- Test Phase ---
        model.eval()
        with torch.no_grad():
            if isItLabel:
                outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
            else:
                outputs = model(X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_combined)[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_StreamKMpp_Cosine_GSL_add_ver2_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    #memory = StreamKMpp_Cosine(k=args.memory, buffer_size=args.memory, coreset_size=args.memory, device=device)
    memory = StreamKMpp_Cosine(buffer_size=args.memory, device=device)

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "GCN":
        model = Simple_GSL_add_ver2(args).to(device) # Simple_GSL(args).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = feature_buffer_label
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory))
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else:
            temp = memory.get_centroids()
            if temp is None:
                if len(memory.buffer)==0:
                    X_combined = torch.cat((feature_buffer_label, input))
                else:
                    buffered_memory = torch.stack(memory.buffer).to(device)
                    X_combined = torch.cat((feature_buffer_label, buffered_memory, input))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
            memory.partial_fit(input.squeeze(0))
        # --- Test Phase ---
        model.eval()
        with torch.no_grad():
            if isItLabel:
                outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
            else:
                outputs = model(X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_combined)[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_random_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = args.memory

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "MLP":
        model = SimpleMLP(dataset_name).to(device)
    elif encoder_name == "GCN":
        model = SimpleGCN(dataset_name).to(device)
    elif encoder_name == "GraphSage":
        model = SimpleGraphSage(dataset_name).to(device)
    elif encoder_name == "GAT":
        model = SimpleGAT(dataset_name).to(device)
    elif encoder_name == "TAG":
        model = SimpleTAG(dataset_name).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
        else:
            feature_buffer_unlabel = torch.cat((feature_buffer_unlabel, input))

        if len(feature_buffer_unlabel)>memory:
            N = len(feature_buffer_unlabel)
            idx_to_remove = torch.randint(0, N - 1, (1,)).item()
            feature_buffer_unlabel = torch.cat((feature_buffer_unlabel[:idx_to_remove], feature_buffer_unlabel[idx_to_remove+1:]), dim=0)
            if idx_to_remove == memory:
                print(N, idx_to_remove, len(feature_buffer_unlabel))

        X_combined = torch.cat((feature_buffer_label, feature_buffer_unlabel))
    
        if encoder_name == "MLP":
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(X_combined)[-1].view(1, -1)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            graph = generate_kNN_Setting1(X_combined, nearest_k, direction=direction)
            
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(graph, X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(graph, X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0:
            print(f"{len(feature_buffer_label)} labeled, {len(feature_buffer_unlabel)} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
            if encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
                print(graph)
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        if encoder_name == "MLP":
            loss = criterion(model(feature)[:len(feature_buffer_label)], labels)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            loss = criterion(model(graph, X_combined)[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_out_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = args.memory

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "MLP":
        model = SimpleMLP(dataset_name).to(device)
    elif encoder_name == "GCN":
        model = SimpleGCN(dataset_name).to(device)
    elif encoder_name == "GraphSage":
        model = SimpleGraphSage(dataset_name).to(device)
    elif encoder_name == "GAT":
        model = SimpleGAT(dataset_name).to(device)
    elif encoder_name == "TAG":
        model = SimpleTAG(dataset_name).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
        else:
            if len(feature_buffer_unlabel)==memory:
                #temp_X_combined = torch.cat((feature_buffer_label, feature_buffer_unlabel[:-1]))
                #temp_graph = generate_kNN_Setting1(temp_X_combined, nearest_k, direction=direction)
                #degrees = temp_graph.out_degrees()
                degrees = graph.out_degrees()
                degrees = degrees[len(feature_buffer_label):]
                if len(degrees)!=memory:
                    print("error")
                _, topk_indices = torch.topk(degrees, 1, largest=False)
                mask = torch.ones(len(feature_buffer_unlabel), dtype=torch.bool, device=device)
                mask[topk_indices]=False
                feature_buffer_unlabel=feature_buffer_unlabel[mask]
                if len(feature_buffer_unlabel)!=(memory-1):
                    print("error")
            feature_buffer_unlabel = torch.cat((feature_buffer_unlabel, input))

        X_combined = torch.cat((feature_buffer_label, feature_buffer_unlabel))
    
        if encoder_name == "MLP":
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(X_combined)[-1].view(1, -1)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            graph = generate_kNN_Setting1(X_combined, nearest_k, direction=direction)
            
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(graph, X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(graph, X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0:
            print(f"{len(feature_buffer_label)} labeled, {len(feature_buffer_unlabel)} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
            if encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
                print(graph)
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        if encoder_name == "MLP":
            loss = criterion(model(feature)[:len(feature_buffer_label)], labels)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            loss = criterion(model(graph, X_combined)[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_less_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = args.memory

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "MLP":
        model = SimpleMLP(dataset_name).to(device)
    elif encoder_name == "GCN":
        model = SimpleGCN(dataset_name).to(device)
    elif encoder_name == "GraphSage":
        model = SimpleGraphSage(dataset_name).to(device)
    elif encoder_name == "GAT":
        model = SimpleGAT(dataset_name).to(device)
    elif encoder_name == "TAG":
        model = SimpleTAG(dataset_name).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
        else:
            if len(feature_buffer_unlabel)==memory:
                normalized = F.normalize(X_combined, dim=1, p=2)
                similarity = normalized@normalized.T
                #sim = torch.sum(similarity, dim=1)
                values, _ = torch.topk(similarity, k=nearest_k+1, largest=True)
                sim = torch.sum(values, dim=1)
                sim = sim[len(feature_buffer_label):]
                if len(sim) != memory:
                    print(len(feature_buffer_unlabel), len(sim))
                _, topk_indices = torch.topk(sim, 1, largest=False)
    
                mask = torch.ones(len(feature_buffer_unlabel), dtype=torch.bool, device=device)
                mask[topk_indices]=False
                
                feature_buffer_unlabel=feature_buffer_unlabel[mask]
                if len(feature_buffer_unlabel)!=(memory-1):
                    print("error")
            feature_buffer_unlabel = torch.cat((feature_buffer_unlabel, input))

        X_combined = torch.cat((feature_buffer_label, feature_buffer_unlabel))
    
        if encoder_name == "MLP":
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(X_combined)[-1].view(1, -1)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            graph = generate_kNN_Setting1(X_combined, nearest_k, direction=direction)
            
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(graph, X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(graph, X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0:
            print(f"{len(feature_buffer_label)} labeled, {len(feature_buffer_unlabel)} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
            if encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
                print(graph)
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        if encoder_name == "MLP":
            loss = criterion(model(feature)[:len(feature_buffer_label)], labels)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            loss = criterion(model(graph, X_combined)[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_more_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = args.memory

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "MLP":
        model = SimpleMLP(dataset_name).to(device)
    elif encoder_name == "GCN":
        model = SimpleGCN(dataset_name).to(device)
    elif encoder_name == "GraphSage":
        model = SimpleGraphSage(dataset_name).to(device)
    elif encoder_name == "GAT":
        model = SimpleGAT(dataset_name).to(device)
    elif encoder_name == "TAG":
        model = SimpleTAG(dataset_name).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
        else:
            if len(feature_buffer_unlabel)==memory:
                normalized = F.normalize(X_combined, dim=1, p=2)
                similarity = normalized@normalized.T
                #sim = torch.sum(similarity, dim=1)
                values, _ = torch.topk(similarity, k=nearest_k+1, largest=True)
                sim = torch.sum(values, dim=1)
                sim = sim[len(feature_buffer_label):]
                if len(sim) != memory:
                    print(len(feature_buffer_unlabel), len(sim))
                _, topk_indices = torch.topk(sim, 1, largest=True)
    
                mask = torch.ones(len(feature_buffer_unlabel), dtype=torch.bool, device=device)
                mask[topk_indices]=False
                
                feature_buffer_unlabel=feature_buffer_unlabel[mask]
                if len(feature_buffer_unlabel)!=(memory-1):
                    print("error")
            feature_buffer_unlabel = torch.cat((feature_buffer_unlabel, input))

        X_combined = torch.cat((feature_buffer_label, feature_buffer_unlabel))
    
        if encoder_name == "MLP":
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(X_combined)[-1].view(1, -1)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            graph = generate_kNN_Setting1(X_combined, nearest_k, direction=direction)
            
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(graph, X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(graph, X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0:
            print(f"{len(feature_buffer_label)} labeled, {len(feature_buffer_unlabel)} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
            if encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
                print(graph)
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        if encoder_name == "MLP":
            loss = criterion(model(feature)[:len(feature_buffer_label)], labels)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            loss = criterion(model(graph, X_combined)[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_more_all_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = args.memory

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "MLP":
        model = SimpleMLP(dataset_name).to(device)
    elif encoder_name == "GCN":
        model = SimpleGCN(dataset_name).to(device)
    elif encoder_name == "GraphSage":
        model = SimpleGraphSage(dataset_name).to(device)
    elif encoder_name == "GAT":
        model = SimpleGAT(dataset_name).to(device)
    elif encoder_name == "TAG":
        model = SimpleTAG(dataset_name).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
        else:
            if len(feature_buffer_unlabel)==memory:
                normalized = F.normalize(X_combined, dim=1, p=2)
                similarity = normalized@normalized.T
                sim = torch.sum(similarity, dim=1)
                #values, _ = torch.topk(similarity, k=nearest_k+1, largest=True)
                #sim = torch.sum(values, dim=1)
                sim = sim[len(feature_buffer_label):]
                if len(sim) != memory:
                    print(len(feature_buffer_unlabel), len(sim))
                _, topk_indices = torch.topk(sim, 1, largest=True)
    
                mask = torch.ones(len(feature_buffer_unlabel), dtype=torch.bool, device=device)
                mask[topk_indices]=False
                
                feature_buffer_unlabel=feature_buffer_unlabel[mask]
                if len(feature_buffer_unlabel)!=(memory-1):
                    print("error")
            feature_buffer_unlabel = torch.cat((feature_buffer_unlabel, input))

        X_combined = torch.cat((feature_buffer_label, feature_buffer_unlabel))
    
        if encoder_name == "MLP":
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(X_combined)[-1].view(1, -1)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            graph = generate_kNN_Setting1(X_combined, nearest_k, direction=direction)
            
            # --- Test Phase ---
            model.eval()
            with torch.no_grad():
                if isItLabel:
                    outputs = model(graph, X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
                else:
                    outputs = model(graph, X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0:
            print(f"{len(feature_buffer_label)} labeled, {len(feature_buffer_unlabel)} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
            if encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
                print(graph)
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        if encoder_name == "MLP":
            loss = criterion(model(feature)[:len(feature_buffer_label)], labels)
        elif encoder_name in ["GCN", "GraphSage", "GAT", "TAG"]:
            loss = criterion(model(graph, X_combined)[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_GSL_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = args.memory

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "GCN":
        #model = SimpleGCN(dataset_name).to(device)
        model = Simple_GSL(args).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
        else:
            feature_buffer_unlabel = torch.cat((feature_buffer_unlabel, input))

        if len(feature_buffer_unlabel)>memory:
            feature_buffer_unlabel=feature_buffer_unlabel[-memory:]

        X_combined = torch.cat((feature_buffer_label, feature_buffer_unlabel))
    
        # --- Test Phase ---
        model.eval()
        with torch.no_grad():
            if isItLabel:
                outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
            else:
                outputs = model(X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0:
            print(f"{len(feature_buffer_label)} labeled, {len(feature_buffer_unlabel)} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_combined)[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_GSL_APPNP_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = args.memory

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "GCN":
        #model = SimpleGCN(dataset_name).to(device)
        model = Simple_GSL_APPNP(args).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
        else:
            feature_buffer_unlabel = torch.cat((feature_buffer_unlabel, input))

        if len(feature_buffer_unlabel)>memory:
            feature_buffer_unlabel=feature_buffer_unlabel[-memory:]

        X_combined = torch.cat((feature_buffer_label, feature_buffer_unlabel))
    
        # --- Test Phase ---
        model.eval()
        with torch.no_grad():
            if isItLabel:
                outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
            else:
                outputs = model(X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0:
            print(f"{len(feature_buffer_label)} labeled, {len(feature_buffer_unlabel)} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_combined)[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_GSL_APPNP_correct_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = args.memory

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "GCN":
        #model = SimpleGCN(dataset_name).to(device)
        model = Simple_GSL_APPNP_correct(args).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
        else:
            feature_buffer_unlabel = torch.cat((feature_buffer_unlabel, input))

        if len(feature_buffer_unlabel)>memory:
            feature_buffer_unlabel=feature_buffer_unlabel[-memory:]

        X_combined = torch.cat((feature_buffer_label, feature_buffer_unlabel))
    
        # --- Test Phase ---
        model.eval()
        with torch.no_grad():
            if isItLabel:
                outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
            else:
                outputs = model(X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0:
            print(f"{len(feature_buffer_label)} labeled, {len(feature_buffer_unlabel)} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_combined)[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_GSL_reg_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = args.memory
    lamb = args.lamb

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "GCN":
        #model = SimpleGCN(dataset_name).to(device)
        model = Simple_GSL(args).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
        else:
            feature_buffer_unlabel = torch.cat((feature_buffer_unlabel, input))

        if len(feature_buffer_unlabel)>memory:
            feature_buffer_unlabel=feature_buffer_unlabel[-memory:]

        X_combined = torch.cat((feature_buffer_label, feature_buffer_unlabel))
    
        # --- Test Phase ---
        model.eval()
        with torch.no_grad():
            if isItLabel:
                outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
            else:
                outputs = model(X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0:
            print(f"{len(feature_buffer_label)} labeled, {len(feature_buffer_unlabel)} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_combined)[:len(feature_buffer_label)], labels)
        loss += lamb * model.reg(X_combined)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_GSL_reg_ver2_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = args.memory
    lamb = args.lamb

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "GCN":
        model = Simple_GSL_binary(args).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
        else:
            feature_buffer_unlabel = torch.cat((feature_buffer_unlabel, input))

        if len(feature_buffer_unlabel)>memory:
            feature_buffer_unlabel=feature_buffer_unlabel[-memory:]

        X_combined = torch.cat((feature_buffer_label, feature_buffer_unlabel))
    
        # --- Test Phase ---
        model.eval()
        with torch.no_grad():
            if isItLabel:
                outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
            else:
                outputs = model(X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0:
            print(f"{len(feature_buffer_label)} labeled, {len(feature_buffer_unlabel)} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_combined)[:len(feature_buffer_label)], labels)
        loss += lamb * model.reg(X_combined)
        loss.backward()
        optimizer.step()
    return test_accuracies

def SSL_online_kNN_online_update_with_memory_GSL_add_Setting1(loader, dataset_size, args, lr, wd):
    opt_name = args.opt
    dataset_name = args.dataset
    encoder_name = args.encoder
    label_ratio = args.label_ratio
    nearest_k = args.k
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    direction = args.directed
    memory = args.memory

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)  # Initialize all as False
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]  # Select 1% indices
    LABELS[true_indices] = True  # Set selected indices to True
    print(LABELS)

    criterion = nn.CrossEntropyLoss()
    
    if encoder_name == "GCN":
        #model = Simple_GSL(args).to(device)
        model = Simple_GSL_add(args).to(device)
    print(model)
        
    if opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    feature_buffer_label, feature_buffer_unlabel, labels = torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel:
            feature_buffer_label = torch.cat((feature_buffer_label, input))
            labels = torch.cat((labels, label))
        else:
            feature_buffer_unlabel = torch.cat((feature_buffer_unlabel, input))

        if len(feature_buffer_unlabel)>memory:
            feature_buffer_unlabel=feature_buffer_unlabel[-memory:]

        X_combined = torch.cat((feature_buffer_label, feature_buffer_unlabel))
    
        # --- Test Phase ---
        model.eval()
        with torch.no_grad():
            if isItLabel:
                outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1)
            else:
                outputs = model(X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label.to(device)).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0:
            print(f"{len(feature_buffer_label)} labeled, {len(feature_buffer_unlabel)} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_combined)[:len(feature_buffer_label)], labels)
        loss.backward()
        optimizer.step()
    return test_accuracies

def online_SSL_Match_Setting1(loader, dataset_size, args, lr, wd):
    opt_name, dataset_name, label_ratio, memory_constant, lamb = args.opt, args.dataset, args.label_ratio, args.memory_constant, args.lamb
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "CPU")
    time_decay_tau = args.time_decay_tau
    memory_update_version = "ver2" if args.memory_update_ver2 else "ver1"
    
    unlabeled_memory_size = args.memory
    if memory_constant:
        labeled_memory_size = args.labeled_size
        labeled_memory_type = args.labeled_memory_type 
        unlabeled_memory_size -= labeled_memory_size
    else: # full
        labeled_memory_size = 0
        
    if args.memory_type == "StreamKMpp_Cosine":
        memory = StreamKMpp_Cosine(buffer_size=unlabeled_memory_size, device=device)
    elif args.memory_type == "Update":
        memory = Update(buffer_size=unlabeled_memory_size, device=device)
    elif args.memory_type == "Update_ver2":
        memory = Update_ver2(buffer_size=unlabeled_memory_size, device=device)
    elif args.memory_type == "Update_min":
        memory = Update_min(buffer_size=unlabeled_memory_size, device=device)
    elif args.memory_type == "Update_topk":
        memory = Update_topk(buffer_size=unlabeled_memory_size, device=device, k=nearest_k)
    elif args.memory_type == "Window":
        memory = Window(buffer_size=unlabeled_memory_size, device=device, isLabel=False)
        
    if labeled_memory_type == "Window":
        labeled_memory = Window(buffer_size=labeled_memory_size, device=device)
    elif labeled_memory_type == "Window_time_decaying":
        labeled_memory = Window_time_decaying(buffer_size=labeled_memory_size, device=device)
        
    method = args.method
    aug_weak, aug_strong = args.aug_weak, args.aug_strong
    tau = args.tau

    LABELS = torch.zeros(dataset_size, dtype=torch.bool)
    true_indices = torch.randperm(dataset_size)[:int(label_ratio * dataset_size)]
    LABELS[true_indices] = True
    print(torch.sum(LABELS), LABELS)

    if "time_decaying" in labeled_memory_type:
        criterion = nn.CrossEntropyLoss(reduction='none')
        criterion2 = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    model = Simple_Match(dataset_name, args.layer).to(device) #Simple_GSL_bn(args).to(device)
    print(model)
        
    if opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        extra_args = optional if dataset_name == "Camelyon17" else []
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        if isItLabel: # Labeled memory update
            labeled_memory.partial_fit(feat=input.squeeze(0), label=label.squeeze(0))

            feature_buffer_label, labels = labeled_memory.get_centroids()
            temp = memory.get_centroids()
            if temp is None:
                if "Update" in args.memory_type:
                    X_combined = feature_buffer_label
                else:
                    X_combined = feature_buffer_label if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer))) 
            else:
                X_combined = torch.cat((feature_buffer_label, temp))
        else: # Unlabeled memory update
            feature_buffer_label, labels = labeled_memory.get_centroids()
            temp = memory.get_centroids()
            if temp is None:
                if "Update" in args.memory_type:
                     X_combined = torch.cat((feature_buffer_label, input))
                else:
                    X_combined = torch.cat((feature_buffer_label, input)) if len(memory.buffer)==0 else torch.cat((feature_buffer_label, torch.stack(memory.buffer), input))
                memory.partial_fit(input.squeeze(0))
            else:
                X_combined = torch.cat((feature_buffer_label, temp, input))
                if memory_update_version == "ver2":
                    model.eval()
                    with torch.no_grad():
                        memory_emb = model.get_embedding(temp)
                        input_emb = model.get_embedding(input)
                    memory.partial_fit_ver2(input, input_emb, memory_emb)
                else:
                    memory.partial_fit(input.squeeze(0))
        
        # --- Test Phase ---
        model.eval()
        with torch.no_grad():
            outputs = model(X_combined)[:len(feature_buffer_label)][-1].view(1, -1) if isItLabel else model(X_combined)[-1].view(1, -1)
                
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label).float().mean().item()
        test_accuracies.append(accuracy)
    
        if idx % 10000 == 0 and idx != 0:
            print(f"{len(feature_buffer_label)} labeled, {len(memory.get_centroids())} unlabeled, {idx}-th element, accumulative test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        model.train()
        optimizer.zero_grad()
        ########### Match series
        ########### Augmentation
        X_combined_weak = X_combined + aug_weak * torch.randn_like(X_combined)
        X_combined_strong = X_combined + aug_strong * torch.randn_like(X_combined)
    
        logits_weak = model(X_combined_weak)
        logits_strong = model(X_combined_strong)

        ######## Supervised Loss
        if "time_decaying" in labeled_memory_type:
            loss_temp = criterion(logits_weak[:len(feature_buffer_label)], labels)
            loss = torch.mean(loss_temp * torch.exp(labeled_memory.get_time_decaying()/time_decay_tau))
        else:
            loss = criterion(logits_weak[:len(feature_buffer_label)], labels)
            
        ######## Regularization Loss
        if method == "FixMatch":
            probs = torch.softmax(logits_weak, dim=-1)
            max_probs, pseudo_labels = torch.max(probs, dim=-1)
            mask = max_probs.ge(tau).float()
            unsup_loss = (F.cross_entropy(logits_strong, pseudo_labels, reduction='none') * mask).mean()
            loss += lamb * unsup_loss

        ########
        loss.backward()
        optimizer.step()
    return test_accuracies