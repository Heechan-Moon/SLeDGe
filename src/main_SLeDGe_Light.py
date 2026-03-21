import torch
import pickle
import random
import numpy as np
import os
import json
from tqdm import tqdm
from itertools import product

from helper import main_SLeDGe_Light
from utils import dataloader_online, get_parser

args = get_parser()

dataset_name, label_ratio, time, gpu  = args.dataset, args.label_ratio, args.time, args.gpu

lrs = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
wds = [0.0, 5e-4]
lambs=[0.001, 0.01, 0.1, 1.0]

seeds = [0, 2000, 4000, 6000, 8000]

loader, dataset_size = dataloader_online(dataset_name)

with open(f'config/best_config_{dataset_name}_{label_ratio}.json', 'r') as f:
    best_config = json.load(f)
k, gcn_layer, scorer_layer = best_config['k'], best_config['gcn_layer'], best_config['scorer_layer']

for lr, wd, lamb in product(lrs, wds, lambs):
    for seed in tqdm(seeds):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
            
        directory = f'SLeDGe_Light ({dataset_name}, {label_ratio}, {time})/{k}_k_{gcn_layer}_GCN_{scorer_layer}_scorer/'
        file_name = f'result_{lr}_{wd}_{lamb}_{seed}.pkl'
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        if os.path.exists(f'{directory}/{file_name}'):
            continue

        print(f"================ lr={lr}, wd={wd}, seed={seed} ================")
        test_accuracies = main_SLeDGe_Light(loader, dataset_size, dataset_name, label_ratio, time, gpu, k, gcn_layer, scorer_layer, lr, wd, lamb)
        print(f"Avg Final Test ACC: {np.mean(test_accuracies):.4f}")
        print("==="*100)
        with open(f'{directory}/{file_name}', 'wb') as file:
            pickle.dump(test_accuracies, file)

