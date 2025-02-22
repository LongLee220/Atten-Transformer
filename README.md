# Atten-LSTM: Temporal Attention-Based LSTM for App Usage Prediction

This project implements the Atten-LSTM model, a temporal attention-based LSTM framework designed to improve app usage prediction by integrating time encoding and traffic-aware features.

<img src=Atten_model.jpg width=900>

---

## Table of Contents
1. [Overview](#overview)
2. [Environment Requirements](#environment-requirements)
3. [Installation Instructions](#installation-instructions)
4. [Data](#data)
5. [Preprocessing](#dataset processed)
6. [Run](#run)
7. [Result](#result)

---

## Overview

Accurately predicting app usage patterns can significantly enhance user experience and system performance by preloading app-related resources, reducing startup latency, and improving energy efficiency. Atten-LSTM combines a Long Short-Term Memory (LSTM) network with a temporal attention mechanism to prioritize the most informative app usage sequences, significantly improving prediction accuracy.

---

## Environment Requirements

This project requires the following environment and dependencies to be installed:

### Base Environment
- **Operating System:** Ubuntu 22.04
- **Python Version:** 3.10.12
- **CUDA Version:** 11.8.0 (optional for GPU support)

### Python Dependencies
Below are the required Python packages and their versions:

| Package         | Version    |
|-----------------|------------|
| `torch`         | Compatible with CUDA 11.8 (installed from [PyTorch official index](https://download.pytorch.org/whl/cu118)) |
| `torchvision`   | Compatible with CUDA 11.8 |
| `torchaudio`    | Compatible with CUDA 11.8 |
| `logzero`       | 1.7.0      |
| `networkx`      | 3.1        |
| `pandas`        | 2.0.3      |
| `pykan`         | 0.2.8      |
| `pyyaml`        | 6.0.2      |
| `scikit-learn`  | 1.3.2      |
| `matplotlib`    | 3.8.0      |

---

## Installation Instructions

To set up the environment locally, follow these steps:

### 1. Install Python and CUDA
Ensure you have Python 3.10.12 installed on your system. If GPU acceleration is needed, make sure CUDA 11.8 is installed and configured properly.

### 2. Install Required Python Dependencies
Run the following command to install all necessary Python packages:

```bash
pip install -U \
    logzero==1.7.0 \
    networkx==3.1 \
    pandas==2.0.3 \
    pykan==0.2.8 \
    pyyaml==6.0.2 \
    scikit-learn==1.3.2 \
    matplotlib==3.8.0
```
### 3. Install PyTorch with CUDA 11.8 Support


## Dataset

[Tsinghua App Usage Dataset](http://fi.ee.tsinghua.edu.cn/appusage/)


```
@article{yu2018smartphone,
    title={Smartphone app usage prediction using points of interest},
    author={Yu, Donghan and Li, Yong and Xu, Fengli and Zhang, Pengyu and Kostakos, Vassilis},
    journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
    volume={1},
    number={4},
    pages={174},
    year={2018},
    publisher={ACM}}
```
Extract `App_usage_trace.txt` and put in in the `./Tsinghua/data` directory.


[LSApp Dataset](https://github.com/aliannejadi/LSApp)
```
@inproceedings{AliannejadiTOIS21,
    author    = {Aliannejadi, Mohammad and Zamani, Hamed and Crestani, Fabio and Croft, W. Bruce},
    title     = {Context-Aware Target Apps Selection and Recommendation for Enhancing Personal Mobile Assistants},
    booktitle = {{ACM} Transactions on Information Systems ({TOIS})},
    year      = 2021
  }
```


## Preprocessing
If you want to change sequence length, change `--seq_length` in main.py.
If you want to change split method, change `--split` in main.py.


## Run

After extracting dataset and preprocessing,

```bash
python main.py
```

## Results

### Tsinghua Dataset (Time-based Split)

| **Methods**       | **HR@1** | **HR@2** | **HR@3** | **HR@4** | **HR@5** | **NDCG@1** | **NDCG@2** | **NDCG@3** | **NDCG@4** | **NDCG@5** | **MRR@1** | **MRR@2** | **MRR@3** | **MRR@4** | **MRR@5** |
|--------------------|----------|----------|----------|----------|----------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| **MRU**           | 0.2398   | 0.4196   | 0.5135   | 0.5670   | 0.5938   | 0.2398      | 0.3501      | 0.3921      | 0.4163      | 0.4299      | 0.2398      | 0.3210      | 0.3522      | 0.3668      | 0.3754      |
| **MFU**           | 0.2472   | 0.4205   | 0.5703   | 0.5621   | 0.5987   | 0.2472      | 0.3565      | 0.3999      | 0.4235      | 0.4377      | 0.2472      | 0.3338      | 0.3628      | 0.3765      | 0.3838      |
| **BPRMF**         | 0.3293   | 0.4437   | 0.5188   | 0.5681   | 0.6077   | 0.3293      | 0.4015      | 0.4390      | 0.4602      | 0.4755      | 0.3293      | 0.3865      | 0.4115      | 0.4238      | 0.4317      |
| **GRU3Rec**       | 0.3137   | 0.4493   | 0.5425   | 0.6028   | 0.6494   | 0.3137      | 0.3993      | 0.4459      | 0.4827      | 0.5012      | 0.3137      | 0.3815      | 0.4126      | 0.4277      | 0.4370      |
| **AppUsage2Vec**  | 0.3333   | 0.4592   | 0.5436   | 0.6080   | 0.6560   | 0.3333      | 0.4127      | 0.4549      | 0.4827      | 0.5012      | 0.3333      | 0.3962      | 0.4244      | 0.4405      | 0.4501      |
| **SR-GNN**        | 0.3342   | 0.4716   | 0.5563   | 0.6154   | 0.6626   | 0.3342      | 0.4209      | 0.4632      | 0.4887      | 0.5070      | 0.3342      | 0.4029      | 0.4311      | 0.4459      | 0.4554      |
| **DUGN**          | 0.3479   | 0.4768   | 0.5593   | 0.6215   | 0.6710   | 0.3479      | 0.4292      | 0.4705      | 0.4973      | 0.5164      | 0.3479      | 0.4124      | 0.4399      | 0.4554      | 0.4653      |
| **Appformer**     | 0.4268   | 0.5550   | 0.6230   | 0.6656   | 0.6960   | 0.4268      | 0.5550      | 0.5979      | 0.6192      | 0.6323      | 0.4268      | 0.4909      | 0.5136      | 0.5242      | 0.5303      |
| **Atten-Transformer** | **0.6400** | **0.7852** | **0.8367** | **0.8663** | **0.8863** | **0.6400** | **0.7316** | **0.7574** | **0.7701** | **0.7779** | **0.6400** | **0.7126** | **0.7230** | **0.7372** | **0.7412** |


### Tsinghua Dataset (Standard and Cold-start Split)

| **Methods**      | **HR@1**  | **HR@3**  | **HR@5**  | **MRR@3** | **MRR@5** | **HR@1**  | **HR@3**  | **HR@5**  | **MRR@3** | **MRR@5** |
|----------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|                    | **Standard Split** →  |  |  |  |  | **Cold Start Split** →  |  |  |  |  |
| **MFU**        | 0.1972 | 0.4288 | 0.5384 | 0.2991 | 0.3241 | 0.1853 | 0.3906 | 0.4943 | 0.2752 | 0.2989 |
| **MRU**        | 0.0000 | 0.5538 | 0.6536 | 0.2585 | 0.2817 | 0.0000 | 0.6406 | 0.7226 | 0.3042 | 0.3234 |
| **Appusage2Vec** | 0.2909 | 0.4822 | 0.5781 | 0.3739 | 0.3958 | - | - | - | - | - |
| **NeuSA**      | 0.4640 | 0.6562 | 0.7286 | 0.5492 | 0.5658 | 0.4433 | 0.6169 | 0.6812 | 0.5206 | 0.5353 |
| **SA-GCN**     | 0.0613 | 0.1882 | 0.2521 | 0.1183 | 0.1331 | - | - | - | - | - |
| **DeepApp**    | 0.2862 | 0.5931 | 0.7075 | 0.4210 | 0.4473 | - | - | - | - | - |
| **DeepPattern** | 0.2848 | 0.5884 | 0.7016 | 0.4185 | 0.4444 | - | - | - | - | - |
| **CoSEM**      | 0.4163 | 0.6682 | 0.7499 | 0.5282 | 0.5469 | 0.3111 | 0.5597 | 0.6525 | 0.4204 | 0.4416 |
| **TimesNet**   | 0.0208 | 0.0480 | 0.0614 | 0.0327 | 0.0358 | 0.0144 | 0.0433 | 0.0647 | 0.0277 | 0.0323 |
| **Transformer** | 0.0262 | 0.0534 | 0.0661 | 0.0383 | 0.0412 | 0.0180 | 0.0461 | 0.0606 | 0.0308 | 0.0337 |
| **FEDformer**  | 0.0159 | 0.0420 | 0.0553 | 0.0272 | 0.0303 | 0.0100 | 0.0411 | 0.0610 | 0.0231 | 0.0282 |
| **DLinear**    | 0.0072 | 0.0370 | 0.0607 | 0.0202 | 0.0256 | 0.0070 | 0.0350 | 0.0586 | 0.0199 | 0.0245 |
| **Reformer**   | 0.0228 | 0.0503 | 0.0645 | 0.0346 | 0.0378 | 0.0224 | 0.0506 | 0.0570 | 0.0349 | 0.0384 |
| **MAPLE**      | 0.5191 | **0.7385** | **0.8115** | **0.6169** | **0.6338** | 0.5228 | 0.7417 | 0.8128 | 0.6206 | 0.6369 |
| **Atten-Transformer** | **0.5238** | 0.7236 | 0.7893 | 0.6129 | 0.6280 | **0.6024** | **0.7772** | **0.8240** | **0.6816** | **0.6924** |

### LSapp Dataset (Standard and Cold-start Split)

| **Methods**         | **HR@1** | **HR@3** | **HR@5** | **MRR@3** | **MRR@5** | **HR@1** | **HR@3** | **HR@5** | **MRR@3** | **MRR@5** |
|---------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
|                    | **Standard Split** →  |  |  |  |  | **Cold Start Split** →  |  |  |  |  |
| **MFU**            | 0.2952 | 0.6258 | 0.7942 | 0.4378 | 0.4765 | 0.2952 | 0.6258 | 0.7942 | 0.4378 | 0.4765 |
| **MRU**            | 0.0276 | 0.7850 | 0.8306 | 0.3974 | 0.4079 | 0.0276 | 0.7850 | 0.8306 | 0.3974 | 0.4079 |
| **Appusage2Vec**   | 0.6057 | 0.7858 | 0.8618 | 0.6848 | 0.7022 | 0.6057 | 0.7858 | 0.8618 | 0.6848 | 0.7022 |
| **TimesNet**       | 0.4805 | 0.6280 | 0.6897 | 0.5459 | 0.5600 | 0.4805 | 0.6280 | 0.6897 | 0.5459 | 0.5600 |
| **Transformer**    | 0.4978 | 0.6530 | 0.7141 | 0.5659 | 0.5800 | 0.4978 | 0.6530 | 0.7141 | 0.5659 | 0.5800 |
| **FEDformer**      | 0.4946 | 0.6374 | 0.6915 | 0.5585 | 0.5708 | 0.4946 | 0.6374 | 0.6915 | 0.5585 | 0.5708 |
| **DLinear**        | 0.1611 | 0.3978 | 0.4790 | 0.2637 | 0.2824 | 0.1611 | 0.3978 | 0.4790 | 0.2637 | 0.2824 |
| **Reformer**       | 0.4920 | 0.6505 | 0.7074 | 0.5620 | 0.5750 | 0.4920 | 0.6505 | 0.7074 | 0.5620 | 0.5750 |
| **CoSEM**          | 0.4990 | 0.7466 | 0.8149 | 0.6083 | 0.6242 | 0.4990 | 0.7466 | 0.8149 | 0.6083 | 0.6242 |
| **NeuSA**          | 0.6832 | 0.8253 | 0.8830 | 0.7461 | 0.7593 | 0.6832 | 0.8253 | 0.8830 | 0.7461 | 0.7593 |
| **MAPLE**          | 0.7157 | 0.8649 | 0.9150 | 0.7821 | 0.7936 | 0.7171 | 0.8670 | 0.9166 | 0.7836 | 0.7950 |
| **Atten-Transformer** | **0.8115** | **0.9494** | **0.9667** | **0.8767** | **0.8807** | **0.8480** | **0.9662** | **0.9807** | **0.9042** | **0.9075** |
