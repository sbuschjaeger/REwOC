
# Rejection Ensembles with Online Calibration


This is the code for our ECML-PKDD 2024 Paper ``Rejection Ensembles with Online Calibration''. The general structure of this code is as follows:

- `RejectionEnsemble.py` contains the implementation of a Rejection Ensemble trained via virtual-labels and confidence scores for [scikit-learn](https://scikit-learn.org/)-based classifiers. It should work with any [scikit-learn](https://scikit-learn.org/) classifier, but does not implement the proper [scikit-learn](https://scikit-learn.org/) interface at this moment. However it offers the typical `fit` and `predict_proba` methods.
- `RejectionEnsembleTorch.py` contains the implementation of a Rejection Ensemble trained via virtual-labels and confidence scores for [PyTorch](https://pytorch.org/)-based classifiers. It should work with most [PyTorch](https://pytorch.org/) classifiers, but we only tested a handfull torchvision models at this moment. It also offers the typical `fit` and `predict_proba` methods and returns numpy arrray (not `torch.tensor`)
-  `run_torch.py`: This script runs the experiments on CIFAR100 and ImageNet. For detailed description please see the argument list.
-  `run_uci.py`: This script runs the experiments on the UCI Datasets. For detailed description please see the argument list.
-  `utils.py`: This file contains some helper functions. 
-  `Datasets.py`: This file contains some helper functions to automatically download and pre-process UCI datasets. 
-  `plot.ipynb`: This Jupyter notebook can be used to generate the plots for the paper. Please see the additional comments in this file on how to use it.


# Running the experiments


## Generate a virtual environment 

It is recommended to generate a virtual environment and install all dependencies into it. **Note**: For the jetson boards there is an additional `requirements_jetson.txt` file that contains `jetson-stats` as well as custom `torch` and `torchvision` packages. If you want to run these experiments on a Jetson board, please use these requirements and adapt the pathes to `torch` and `torchvision` accordingly.

```
mkdir -P .venv/rewoc && python -m venv .venv/rewoc
source .venv/rewoc/bin/activate
pip install -r requirements.txt
```

## Running the experiments

Running the UCI experiments is straight forward, as the data will automatically be downloaded. Inside the environment just execute

```
./run_uci.py -x 5 -p 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 -M 32 --rejector linear --small dt --big rf --data weather covtype eeg elec gas-drift anuran --tmp ./data/ --out rf
./run_uci.py -x 5 -p 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 -M 32 --rejector linear --small dt --big dt3 --data weather covtype eeg elec gas-drift anuran --tmp ./data/ --out dt3
``` 
to reproduce the experiments in the paper. If you want to additionally measure the energy consumption on Jetson boards, add the `-e` parameter. 

Running the CIFAR100 and ImageNet experiments is also straight forward. However note, that the ImageNet data will not be automatically downloaded. Hence, you have to provide the path to the ImagNet validation data manually:  

```
./run_torch.py -x 5 -p 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 -M 32 --rejector linear --small cifar100_shufflenetv2_x1_0 --data cifar100 --tmp ./data/
./run_torch.py -x 5 -p 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 -M 32  --rejector linear --small shufflenet_v2_x1_0  --big efficientnet_b4 --data imagenet --tmp /path/to/imagenet/
``` 

## Cite the paper

```
@InProceedings{Buschjaeger/2024,
  abstract           = {As machine learning models become increasingly integrated into various applications, the need for resource-aware deployment strategies becomes paramount. One promising approach for optimizing resource consumption is rejection ensembles.  Rejection ensembles combine a small model deployed to an edge device with a large model deployed in the cloud with a rejector tasked to determine the most suitable model for a given input. Due to its novelty, existing research predominantly focuses on ad-hoc ensemble design, lacking a thorough understanding of rejector optimization and deployment strategies. This paper addresses this research gap by presenting a theoretical investigation into rejection ensembles and proposing a novel algorithm for training and deploying rejectors based on these novel insights. We give precise conditions of when a good rejector can improve the ensemble's overall performance beyond the big model's performance and when a bad rejector can make the ensemble worse than the small model. Second, we show that even the perfect rejector can overuse its budget for using the big model during deployment. Based on these insights, we propose to ignore any budget constraints during training but introduce additional safeguards during deployment. Experimental evaluation on 8 different datasets from various domains demonstrates the efficacy of our novel rejection ensembles outperforming existing approaches. Moreover, compared to standalone large model inference, we highlight the energy efficiency gains during deployment on a Nvidia Jetson AGX board.},
  author             = {Buschjäger, Sebastian},
  booktitle          = {European Conference on Machine Learning and Knowledge Discovery in Databases, ECML PKDD},
  title              = {Rejection Ensembles with Online Calibration (RewOC) (to appear)},
  year               = {2024},
  code               = {https://github.com/sbuschjaeger/rewoc}
}
```