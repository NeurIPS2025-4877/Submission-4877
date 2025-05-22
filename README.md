# Treatment-Conditioned Hypernetwork for Generalizable Causal Effect Estimation

## Introduction
This repository contains source code for the NeurIPS 2025 submission paper "Treatment-Conditioned Hypernetwork for Generalizable Causal Effect Estimation."

Real-world data provide a valuable resource for estimating conditional average treatment effects (CATE), a key task in personalized medicine. However, limited data availability—especially for rare or newly introduced treatments—poses challenges for reliable and generalizable estimation. This motivates the need for a unified modeling approach that generalizes across multiple treatments. While existing studies leverage treatment information for unified modeling, most rely on a shared architecture across all treatments, which fails to capture treatment-specific covariate relevance and interaction effects. We propose HyperTE, a novel treatment-conditioned hypernetwork framework for CATE estimation that explicitly models treatment–specific covariate interactions and their effects on outcomes. HyperTE leverages treatment attributes to generate the weights of an interaction model via a hypernetwork, enabling covariate representations to be dynamically conditioned on treatment properties. It supports flexible treatment-specific modeling while facilitating knowledge sharing across treatments, enhancing generalization to rare or unseen treatments. To address confounding in the presence of treatment–covariate interactions, we introduce a double-adjustment that decomposes the effects of covariates and interactions for robust causal inference. Extensive experiments demonstrate the effectiveness of our framework in estimating CATE across diverse treatments and generalizing to previously unseen treatments. 


## Overview
![image](https://github.com/user-attachments/assets/99212a16-0549-483e-bcaa-10c222001d20)
Figure 1: Overview of the study problem. (a) Each individual has covariates (X), structured treatments (T), and an observed outcome (Y). (b) The causal graph modeled with $M$ that captures treatment–covariate interactions. (c) The treatment-specific model (no weight sharing) and the treatment-agnostic model (hard weight sharing). The width of the lines represents the magnitude of weights.



## Installation
Our model depends on Numpy, and PyTorch (CUDA toolkit if use GPU). You must have them installed before using our model
>
* Python 3.9
* Pytorch 1.10.2
* Numpy 1.21.2
* Pandas 1.4.1

This command will install all the required libraries and their specified versions.
```python 
pip install -r requirements.txt
```

## Data preparation
### get datasets
The downloadable version of the datasets used in the paper can be accessed in the 'simulation' folder. 

_Note: ._

---

## Training and test
### Python command
For training and evaluating the model, run the following code
```python 
# Note 1: hyper-parameters are included in config.json.
# Note 2: the code simulates the data.
python train.py --config 'config.json' --data 'TCGA'
```
  
### Parameters
Hyper-parameters are set in train.py
>
* `data`: dataset to use; {'TCGA', 'CCLE', 'GDSC'}.
* `config`: .json file

Hyper-parameters are set in *.json
>
* `train_ratio`: the ratio of training
* `drug_n_dims`: the hidden dimension of the treatment embedding layers.
* `drug_n_layers`: the number of layers in the treatment embedding.
* `feat_n_dims`: the hidden dimension of the covariate embedding layers.
* `feat_n_layers`: the number of layers in the covariate embedding.
* `pred_n_dims`: the hidden dimension of the prediction layers.
* `pred_n_layers`: the number of layers in the prediction layers.
* `metrics`: metrics to print out. It is a list format. Functions for all metrics should be included in 'model/metric.py'.
* `early_stop`: the number of epochs for early stopping
* `monitor`: the criterion for early stopping. The first word is 'min' or 'max', and the second one is metric.


_* Experiments were conducted using a computing cluster consisting of 42 nodes, each equipped with dual Intel Xeon 8268 processors, 384GB RAM, and dual NVIDIA Volta V100 GPUs with 32GB memory._


## Results
Table 1: Comparison of performances on different K treatments, measured by WPEHE@K.
<div align="center">
<img src="https://github.com/user-attachments/assets/434962fc-5779-48c3-9517-93b418b4b40b" width="600px">
</div>


Table 2: Zero-shot evaluation compared to treatment-specific models, measured by ePEHE and $e$ATE.
<div align="center">
<img src="https://github.com/user-attachments/assets/955cbd6c-767e-455c-9ae2-b12081f0b22f" width="300px">
</div>



