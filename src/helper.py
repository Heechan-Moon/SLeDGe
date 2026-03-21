import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

from utils import *
from Memory import *

def main_SLeDGe(loader, dataset_size, dataset_name, label_ratio, time, gpu, k, gcn_layer, scorer_layer, lr, wd, lamb):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "CPU")
    n_input, n_classes = dataset_adaptation(dataset_name)
    
    if label_ratio == 0.01:
        labeled_size = 100
    elif label_ratio == 0.001:
        labeled_size = 10
        
    memory = Update_adaptive_EMA(n_classes=n_classes, feat_dim=n_input, labeled_buffer_size=max(n_classes, labeled_size), unlabeled_buffer_size=1000-max(n_classes, labeled_size), device=device)
    print(memory)
    
    LABELS = build_labels_from_single_batch_loader(loader, dataset_size, label_ratio, n_classes, device)
    print(torch.sum(LABELS), LABELS)

    model = model_SLeDGe(gcn_layer, n_input, n_classes, scorer_layer, k).to(device)
    print(model)

    if time:
        criterion = nn.CrossEntropyLoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)
        
        isItLabel = LABELS[idx]

        # --- Test Phase ---
        X_combined_test = memory.get_test(input)
        model.eval()
        with torch.no_grad():
            outputs = model(X_combined_test)[-1].view(1, -1)
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label).float().mean().item()
        test_accuracies.append(accuracy)

        ### memory update
        model.eval()
        if isItLabel:
            with torch.no_grad():
                feat_embs = model.embedding_function.internal_forward(input)
                feat_embs = F.normalize(feat_embs, dim=-1)
                
                embs = model.embedding_function.internal_forward(memory.labeled_bank)
                embs = F.normalize(embs, dim=-1)
            
            X_combined, label_combined = memory.partial_fit(feat=input.squeeze(0), feat_embs=feat_embs, embs=embs, label=label.squeeze(0))
        else:
            with torch.no_grad():
                feat_embs = model.embedding_function.internal_forward(input)
                feat_embs = F.normalize(feat_embs, dim=-1)
                
                embs = model.embedding_function.internal_forward(memory.unlabeled_bank)
                embs = F.normalize(embs, dim=-1)
            
            X_combined, label_combined = memory.partial_fit(feat=input.squeeze(0), feat_embs=feat_embs, embs=embs, label=None)

        num_labeled_elements = memory.labeled_filled.sum()
        if idx % 10000 == 0 and idx != 0:
            print(f"X_combined size: {X_combined.shape} \t Activated label: {num_labeled_elements} \t {idx}-th element - accumul. test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        if num_labeled_elements>0:
            model.train()
            optimizer.zero_grad()
            preds = model(X_combined)
            if time:
                loss_temp = criterion(preds[:num_labeled_elements], label_combined)
                loss = torch.mean(loss_temp * torch.exp(-memory.labeled_age[memory.labeled_filled]))
            else:
                loss = criterion(preds[:num_labeled_elements], label_combined)
            if lamb != 0.0:
                loss += lamb * model.reg(X_combined)
            loss.backward()
            optimizer.step()
    return test_accuracies

def main_SLeDGe_Light(loader, dataset_size, dataset_name, label_ratio, time, gpu, k, gcn_layer, scorer_layer, lr, wd, lamb):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "CPU")
    n_input, n_classes = dataset_adaptation(dataset_name)
    
    if label_ratio == 0.01:
        labeled_size = 100
    elif label_ratio == 0.001:
        labeled_size = 10
        
    memory = Update_adaptive_EMA(n_classes=n_classes, feat_dim=n_input, labeled_buffer_size=max(n_classes, labeled_size), unlabeled_buffer_size=1000-max(n_classes, labeled_size), device=device)
    print(memory)
    
    LABELS = build_labels_from_single_batch_loader(loader, dataset_size, label_ratio, n_classes, device)
    print(torch.sum(LABELS), LABELS)

    model = model_SLeDGe_Light(gcn_layer, n_input, n_classes, scorer_layer, k).to(device)
    print(model)

    if time:
        criterion = nn.CrossEntropyLoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                
    test_accuracies = []
    for idx, (input, label, *optional) in enumerate(tqdm(loader, desc="Interleaved test-then-train phase", total=dataset_size)):
        input = input.view(1, -1).to(device)
        input = F.normalize(input, dim=1)
        label = label.view(-1).to(device)

        isItLabel = LABELS[idx]
        
        # --- Test Phase ---
        X_combined_test = memory.get_test(input)
        model.eval()
        with torch.no_grad():
            N = len(X_combined_test)-1
            target_indices_test = torch.tensor([N], device=device)
            outputs = model(X_combined_test, target_indices_test)[target_indices_test].view(1, -1)
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == label).float().mean().item()
        test_accuracies.append(accuracy)

        ### memory update
        model.eval()
        if isItLabel:
            with torch.no_grad():
                feat_embs = model.embedding_function.internal_forward(input)
                feat_embs = F.normalize(feat_embs, dim=-1)
                
                embs = model.embedding_function.internal_forward(memory.labeled_bank)
                embs = F.normalize(embs, dim=-1)
            
            X_combined, label_combined = memory.partial_fit(feat=input.squeeze(0), feat_embs=feat_embs, embs=embs, label=label.squeeze(0))
        else:
            with torch.no_grad():
                feat_embs = model.embedding_function.internal_forward(input)
                feat_embs = F.normalize(feat_embs, dim=-1)
                
                embs = model.embedding_function.internal_forward(memory.unlabeled_bank)
                embs = F.normalize(embs, dim=-1)
            
            X_combined, label_combined = memory.partial_fit(feat=input.squeeze(0), feat_embs=feat_embs, embs=embs, label=None)
        
        num_labeled_elements = memory.labeled_filled.sum()
        if idx % 10000 == 0 and idx != 0:
            print(f"X_combined size: {X_combined.shape} \t Activated label: {num_labeled_elements} \t {idx}-th element - accumul. test ACC: {np.mean(test_accuracies):.4f}")
    
        # --- Train Phase ---
        if num_labeled_elements>0:
            model.train()
            optimizer.zero_grad()
            target_indices = torch.arange(num_labeled_elements, device=device)
            preds = model(X_combined, target_indices)
            if time:
                loss_temp = criterion(preds[:num_labeled_elements], label_combined)
                loss = torch.mean(loss_temp * torch.exp(-memory.labeled_age[memory.labeled_filled]))
            else:
                loss = criterion(preds[:num_labeled_elements], label_combined)
            if lamb != 0.0:
                loss += lamb * model.reg(X_combined, target_indices)
            loss.backward()
            optimizer.step()
    return test_accuracies