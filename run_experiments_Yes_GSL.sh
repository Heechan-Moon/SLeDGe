#!/bin/bash

#DATASET="MNIST"
DATASET="CIFAR-10"
#DATASET="KMNIST"
#DATASET="FashionMNIST"

#DATASET="Shuttle"
#DATASET="GSD"
#DATASET="SLDD"
#DATASET="HAR"
#DATASET="Occupancy"

#DATASET="CR4"
#DATASET="CRE4V2"
#DATASET="FG2C2D"
#DATASET="GEAR2C2D"
#DATASET="MG2C2D"

#DATASET="MNDS_level_1"
#DATASET="WebKB"
#DATASET="online_shopper_intention"

#DATASET="random_600000"

GSL_type="GSL_reg_ver2_bn"
memory="Update"
memory2="Window_time_decaying"

#lambs=(0.001 0.01 0.1 1.0)
lambs=(1.0)

#ks=(1 5 10)
ks=(10 5 1)

tau=100

label_ratio=0.01

#labeled_size=200
labeled_size=100
#labeled_size=50

#mem=1100
#mem=1000
#mem=950

#mem=550
mem=1900

layers=(1 2 3)
#layers=(2)

#gcn_layers=(1 2 3)
gcn_layers=(1)

gpu=4


for k in "${ks[@]}"; do
    for lamb in "${lambs[@]}"; do
        for layer in "${layers[@]}"; do
            for gcn_layer in "${gcn_layers[@]}"; do
                echo "${DATASET} ${k} ${gcn_layer} ${layer} ${lamb} ${memory} ${memory2}"
                python online_SSL_kNN_memory_GSL_Setting1.py --directed --memory_constant --dataset=${DATASET} --memory_type=${memory} --labeled_memory_type=${memory2} --GSL_type=${GSL_type} --gcn_layer=${gcn_layer} --k=${k} --edge_scorer_layer=${layer} --lamb=${lamb} --gpu=${gpu} --label-ratio=${label_ratio} --labeled_size=${labeled_size} --memory=${mem} --time_decay_tau=${tau}
                #python t1.py --directed --memory_constant --dataset=${DATASET} --memory_type=${memory} --labeled_memory_type=${memory2} --GSL_type=${GSL_type} --gcn_layer=${gcn_layer} --k=${k} --edge_scorer_layer=${layer} --lamb=${lamb} --gpu=${gpu} --label-ratio=${label_ratio} --labeled_size=${labeled_size} --memory=${mem} --time_decay_tau=${tau}
                #python t2.py --directed --memory_constant --dataset=${DATASET} --memory_type=${memory} --labeled_memory_type=${memory2} --GSL_type=${GSL_type} --gcn_layer=${gcn_layer} --k=${k} --edge_scorer_layer=${layer} --lamb=${lamb} --gpu=${gpu} --label-ratio=${label_ratio} --labeled_size=${labeled_size} --memory=${mem} --time_decay_tau=${tau}
                #python t3.py --directed --memory_constant --dataset=${DATASET} --memory_type=${memory} --labeled_memory_type=${memory2} --GSL_type=${GSL_type} --gcn_layer=${gcn_layer} --k=${k} --edge_scorer_layer=${layer} --lamb=${lamb} --gpu=${gpu} --label-ratio=${label_ratio} --labeled_size=${labeled_size} --memory=${mem} --time_decay_tau=${tau}
            done
        done
    done
done


