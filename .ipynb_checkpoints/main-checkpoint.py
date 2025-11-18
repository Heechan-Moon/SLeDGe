import torch
import pickle
import random
import numpy as np
import os
from tqdm import tqdm

from helper_semisupervised_Setting1 import online_SSL
from utils import dataloader_online, get_parser

args = get_parser()

opt_name, dataset_name, encoder_name, label_ratio, k, gcn_layer, direction = args.opt, args.dataset, args.encoder, args.label_ratio, args.k, args.gcn_layer, args.directed

memory, memory_type, memory_constant, labeled_memory_type = args.memory, args.memory_type, args.memory_constant, args.labeled_memory_type

labeled_size = args.labeled_size if memory_constant else "Full"

GSL_type, edge_scorer_layer, edge_scorer, sparsifier, processor, lamb = args.GSL_type, args.edge_scorer_layer, args.edge_scorer, args.sparsifier, args.processor, args.lamb
sparsifier_value = args.k if sparsifier == "kNN" else None
time_decay_tau = args.time_decay_tau

lrs = [1e-5, 5e-5, 1e-4, 5e-4]
wds = [0.0, 5e-4]
seeds = [0, 2000, 4000, 6000, 8000]

for lr in lrs:
    for wd in wds:
        for seed in tqdm(seeds):
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
            loader, dataset_size = dataloader_online(dataset_name)
            directory = f'[{direction}] SLeDGe ({dataset_name}) ({gcn_layer} GCN)/GSL_{GSL_type}_{edge_scorer_layer}_{edge_scorer}_{sparsifier_value}_{sparsifier}_{processor}_{lamb}/{encoder_name}_{opt_name}_{dataset_name}_{label_ratio}_label_{k}_kNN_{labeled_size}_{labeled_memory_type}_{time_decay_tau}_{memory}_{memory_type}_{memory_constant}'
            file_name = f'result_{lr}_{wd}_{seed}.pkl'
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            if os.path.exists(f'{directory}/{file_name}'):
                continue

            print(f"================ lr={lr}, wd={wd}, seed={seed} ================")
            test_accuracies = online_SSL(loader, dataset_size, args, lr, wd)
            print(f"Avg Final Test ACC: {np.mean(test_accuracies):.4f}")
            print("==="*100)
            with open(f'{directory}/{file_name}', 'wb') as file:
                pickle.dump(test_accuracies, file)

