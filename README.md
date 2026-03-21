# SLeDGe
SLeDGe (**S**emi-supervised **Le**arning on **D**ata stream with **G**raph structure l**e**arning) is a framework designed for semi-supervised learning on data streams.
It simultaneously learns a predictive model and an adaptive graph structure, specifically optimized for scenarios with scarce labels and limited storage capacity.

## Experiments

### 1. Overview
* **Label Ratio:** Supported ratios are `0.01` or `0.001`.
* **Temporal Weighting:** Append the `-time` flag to enable temporal weighting.

### 2. Running SLeDGe
* **Using the best configuration:**
    ```bash
    python src/main_SLeDGe.py --dataset={dataset_name} --gpu={gpu_id} --label-ratio={label_ratio}
    ```
* **Running all configurations:**
    ```bash
    python src/main_SLeDGe_full.py --dataset={dataset_name} --gpu={gpu_id} --label-ratio={label_ratio}
    ```

### 3. Running SLeDGe_Light
* **Using the best configuration:**
    ```bash
    python src/main_SLeDGe_Light.py --dataset={dataset_name} --gpu={gpu_id} --label-ratio={label_ratio}
    ```
* **Running all configurations:**
    ```bash
    python src/main_SLeDGe_Light_full.py --dataset={dataset_name} --gpu={gpu_id} --label-ratio={label_ratio}
    ```

### 4. Using Temporal Weighting
To apply temporal weighting, add `-time` to the end of any command:
```bash
python src/main_SLeDGe.py --dataset={dataset_name} --gpu={gpu_id} --label-ratio={label_ratio} -time



## Datasets
- The following table summarizes the statistics and sources for the benchmarks used in this framework.

| Datasets | \|f\| | \|c\| | \|D\| | reference |
|---|---:|---:|---:|---|
| MNDS | 500 | 17 | 10,917 |https://github.com/alinapetukhova/mn-ds-news-classification|
| Shopper | 15 | 2 | 12,330 |https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset|
| WebKB | 500 | 7 | 8,282 |https://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/|
| MNIST | 784 | 10 | 70,000 |https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST|
| CIFAR-10 | 3,072 | 10 | 60,000 |https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10|
| KMNIST | 784 | 10 | 70,000 |https://docs.pytorch.org/vision/main/generated/torchvision.datasets.KMNIST.html#torchvision.datasets.KMNIST|
| FashionMNIST | 784 | 10 | 70,000 |https://docs.pytorch.org/vision/main/generated/torchvision.datasets.FashionMNIST.html#torchvision.datasets.FashionMNIST|
| Shuttle | 7 | 7 | 58,000 |https://archive.ics.uci.edu/dataset/148/statlog+shuttle|
| GSAD | 128 | 6 | 13,910 |https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset|
| SDD | 48 | 11 | 58,509 |https://archive.ics.uci.edu/dataset/325/dataset+for+sensorless+drive+diagnosis|
| HAR | 561 | 6 | 10,299 |https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones|
| OD | 5 | 2 | 20,560 |https://archive.ics.uci.edu/dataset/357/occupancy+detection|

- |f|, |c|, |D| are the number of input features, classes, and data size, respectively.
