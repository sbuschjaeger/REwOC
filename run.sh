#/bin/bash

SECONDS=0

# ./run_uci.py -x 5 -p 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 -e -M 32 --rejector linear --small dt --big rf --data weather covtype eeg elec gas-drift anuran --tmp ./data/ --out rf
# ./run_uci.py -x 5 -p 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 -e -M 32 --rejector linear --small dt --big dt3 --data weather covtype eeg elec gas-drift anuran --tmp ./data/ --out dt3
./run_torch.py -x 5 -p 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 -e -M 32 --rejector linear --small cifar100_shufflenetv2_x1_0 --data cifar100 --tmp ./data/
./run_torch.py -x 5 -p 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 -e -M 32  --rejector linear --small shufflenet_v2_x1_0  --big efficientnet_b4 --data imagenet --tmp /mnt/ssd/data/ImageNet/
DURATION_IN_SECONDS=$SECONDS
echo $DURATION_IN_SECONDS
