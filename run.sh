#!/bin/bash

DATASET="MNIST"
#DATASET="CIFAR-10"
#DATASET="KMNIST"
#DATASET="FashionMNIST"

GSL_type="SLeDGe"
unlabel_memory="Update"
label_memory="Window_time_decaying"

lambs=(0.001 0.01 0.1 1.0)
#lambs=(1.0)

ks=(1 5 10)
#ks=(10 5 1)

tau=100

label_ratio=0.01
labeled_size=100
mem=1000

layers=(1 2 3)

gcn_layers=(1 2 3)

gpu=0


for k in "${ks[@]}"; do
    for lamb in "${lambs[@]}"; do
        for layer in "${layers[@]}"; do
            for gcn_layer in "${gcn_layers[@]}"; do
                echo "${DATASET} ${k} ${gcn_layer} ${layer} ${lamb} ${unlabel_memory} ${label_memory}"
                python src/main.py --directed --memory_constant --dataset=${DATASET} --memory_type=${unlabel_memory} --labeled_memory_type=${label_memory} --GSL_type=${GSL_type} --gcn_layer=${gcn_layer} --k=${k} --embedding_function_layer=${layer} --lamb=${lamb} --gpu=${gpu} --label-ratio=${label_ratio} --labeled_size=${labeled_size} --memory=${mem} --time_decay_tau=${tau}
            done
        done
    done
done


