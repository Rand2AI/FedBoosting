# FedBoosting: Federated Learning with Gradient Protected Boosting for Text Recognition

## Introduction

This is the implementation of the paper "FedBoosting: Federated Learning with Gradient Protected Boosting for Text Recognition". We show in this paper that the generalization ability of the joint model is poor on Non-Independent and Non-Identically Distributed (Non-IID) data, particularly when the Federated Averaging (FedAvg) strategy is used due to the weight divergence phenomenon. We propose a novel boosting algorithm for FL to address this generalization issue, as well as achieving a much faster convergence rate in gradient-based optimization. In addition, a secure gradient sharing protocol using Homomorphic Encryption (HE) and Differential Privacy (DP) is introduced to defend against gradient leakage attack. We demonstrate the proposed Federated Boosting (FedBoosting) method achieves significant improvements in both prediction accuracy and run-time efficiency on text recognition task using public benchmark.

<div align=center><img src="https://github.com/Rand2AI/FedBoosting/blob/main/Image/FedBoost_illustration.png" width=600/></div>

## Requirements

    python==3.6.9

    Flask==2.0.0

    Pillow==7.0.0

    requests==2.23.0

    tensorflow-gpu==1.14.0

    tqdm==4.44.1

    swiss_army_tensorboard (https://github.com/gaborvecsei/Swiss-Army-Tensorboard)

    ...

## Performance

<div align=center><img src="https://github.com/Rand2AI/FedBoosting/blob/main/Image/FedBoost_performance.png" width=600/></div>

## How to use

### Prepare your data:

 * Download datasets online respectively and extract them to "./Data/".
    
 * Run the relevant functions in "./DataProcess/encoder.py" to transfer the data to ".json" format.

 * Spread the data and codes to the server and clients.

### Training

 * Change the pathes and hyper-parameters in "./config.json".

 * Run "./FLtrainer_server.py" firstly and then on each client, run "./FLtrainer_client.py" respectively.

## Citation

If you find this work helpful for your research, please cite the following paper:

    @article{REN2023127126,
    title = {FedBoosting: Federated learning with gradient protected boosting for text recognition},
    journal = {Neurocomputing},
    pages = {127126},
    year = {2023},
    issn = {0925-2312},
    doi = {https://doi.org/10.1016/j.neucom.2023.127126},
    url = {https://www.sciencedirect.com/science/article/pii/S0925231223012493},
    author = {Hanchi Ren and Jingjing Deng and Xianghua Xie and Xiaoke Ma and Yichuan Wang},
    }
