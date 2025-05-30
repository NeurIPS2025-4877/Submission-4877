# Treatment-Conditioned Hypernetwork for Generalizable Causal Effect Estimation

## Introduction
This repository contains source code for the NeurIPS 2025 submission paper "Treatment-Conditioned Hypernetwork for Generalizable Causal Effect Estimation."

Real-world data provide a valuable resource for estimating conditional average treatment effects (CATE), a key task in personalized medicine. However, limited data availability—especially for rare or newly introduced treatments—poses challenges for reliable and generalizable estimation. This motivates the need for a unified modeling approach that generalizes across multiple treatments. While existing studies leverage treatment information for unified modeling, most rely on a shared architecture across all treatments, which fails to capture treatment-specific covariate relevance and interaction effects. We propose HyperTE, a novel treatment-conditioned hypernetwork framework for CATE estimation that explicitly models treatment–specific covariate interactions and their effects on outcomes. HyperTE leverages treatment attributes to generate the weights of an interaction model via a hypernetwork, enabling covariate representations to be dynamically conditioned on treatment properties. It supports flexible treatment-specific modeling while facilitating knowledge sharing across treatments, enhancing generalization to rare or unseen treatments. To address confounding in the presence of treatment–covariate interactions, we introduce a double-adjustment that decomposes the effects of covariates and interactions for robust causal inference. Extensive experiments demonstrate the effectiveness of our framework in estimating CATE across diverse treatments and generalizing to previously unseen treatments. 



## Overview
![image](https://github.com/user-attachments/assets/99212a16-0549-483e-bcaa-10c222001d20)
Figure 1: Overview of the study problem. (a) Each individual has covariates (X), structured treatments (T), and an observed outcome (Y). (b) The causal graph modeled with $M$ that captures treatment–covariate interactions. (c) The treatment-specific model (no weight sharing) and the treatment-agnostic model (hard weight sharing). The width of the lines represents the magnitude of weights.

---

## Installation
This command will install all the required libraries and their specified versions.
```python 
pip install -r requirements.txt
```
---

## Data preparation

To preprocess the CCLE or GDSC datasets, move to the "preprocess/" directory and run:
```python 
python generate_data_ccle.py --data 'GDSC'
```
>
* `data`: specifies the dataset to preprocess. Options are 'CCLE' or 'GDSC'.

To preprocess the TCGA dataset, move to the "preprocess/" directory and run:
```python 
python generate_data_tcga.py 
```

All preprocessed datasets will be saved in the "generated_data/" directory.

---

## Training and test
### Python command
For training and evaluating the model, run the following code:
```python 
# Note 1: hyper-parameters are included in config.json.
python train.py --data 'TCGA' --config 'config/'
```
  
### Parameters
**Hyperparameters are set in train.py**
>
* `data`: dataset to use. Options: 'TCGA', 'CCLE', 'GDSC'.
* `config`: path to the directory containing configuration files.

**Hyperparameters are set in config.json**

"data_loader"
>
* `path_to_train`: path to the training data
* `path_to_test`: path to the test data
* `valid_ratio`: the ratio of validation

"hyper_params"
>
* `drug_n_dims`: the hidden dimension of the treatment embedding layers.
* `drug_n_layers`: the number of layers in the treatment embedding.
* `feat_n_dims`: the hidden dimension of the covariate embedding layers.
* `feat_n_layers`: the number of layers in the covariate embedding.
* `pred_n_dims`: the hidden dimension of the prediction layers.
* `pred_n_layers`: the number of layers in the prediction layers.
* `min_test_assignments`: minimum number of K.
* `max_test_assignments`: maximum number of K.

_Note: Additional hyperparameters are defined for configuring the graph neural network modules, and are not listed here._

"metrics" and "trainer"
>
* `metrics`: metrics to print out. It is a list format. Functions for all metrics should be included in 'model/metric.py'.
* `early_stop`: the number of epochs for early stopping
* `monitor`: the criterion for early stopping. The first word is 'min' or 'max', and the second one is metric.

---

## Results
Table 1: Comparison of performances on different K treatments, measured by WPEHE@K.
<div align="center">
<img src="https://github.com/user-attachments/assets/434962fc-5779-48c3-9517-93b418b4b40b" width="600px">
</div>


Table 2: Zero-shot evaluation compared to treatment-specific models, measured by ePEHE and $e$ATE.
<div align="center">
<img src="https://github.com/user-attachments/assets/955cbd6c-767e-455c-9ae2-b12081f0b22f" width="350px">
</div>

<div align="center">
<img src="https://github.com/user-attachments/assets/ea3c3add-7577-4d73-b6b3-5e869eba6a92" width="900px">
</div>
Figure 1: Comparison of chemical similarity (left) and model weight similarity (right). Brighter values indicate higher similarity.


<div align="center">
<img src="https://github.com/user-attachments/assets/dca76834-4299-4c4e-963d-e9d9b21f9a68" width="900px">
</div>
Figure 2: CATE estimation performance under varying ratios of training treatment used, reported as WPEHE@6 for both in-sample and out-of-sample tests.



