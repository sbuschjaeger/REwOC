#/bin/bash

./run_torch.py -x 5 -p 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --rejector linear --small cifar100_shufflenetv2_x1_0 --data cifar100 --tmp .
./run_torch.py -x 5 -p 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --rejector linear --small mobilenet_v3_small --big vit_b_32 --data imagenet --tmp /cephfs_projects/public/ImageNet/