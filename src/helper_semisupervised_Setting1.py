import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

from utils import *
from Memory import *

def online_SSL(loader, dataset_size, args, lr, wd):
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
        
    if args.memory_type == "Update":
        memory = Update(buffer_size=unlabeled_memory_size, device=device)
        
    if labeled_memory_type == "Window_time_decaying":
        labeled_memory = Window_time_decaying(buffer_size=labeled_memory_size, device=device)
        
    GSL_type = args.GSL_type

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
        model = Simple_GSL(args).to(device)
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
                model.eval()
                with torch.no_grad():
                    memory_emb = model.edge_scorer.internal_forward(temp)
                    input_emb = model.edge_scorer.internal_forward(input)
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
        preds = model(X_combined)
        if "time_decaying" in labeled_memory_type:
            loss_temp = criterion(preds[:len(feature_buffer_label)], labels)
            loss = torch.mean(loss_temp * torch.exp(labeled_memory.get_time_decaying()/time_decay_tau))
        else:
            loss = criterion(preds[:len(feature_buffer_label)], labels)
        loss += lamb * model.reg(X_combined)
        loss.backward()
        optimizer.step()
    return test_accuracies

