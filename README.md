# D³PGSA: A Dual-value Deep Deterministic Policy Gradient based on Sample Augmentation for Dynamic Pricing of EV Charging Stations

This repository provides the code, sample data, and experimental results for our paper:

## Overview

This project aims to address the dynamic pricing problem for electric vehicle charging stations (EVCS) under uncertain and dynamic demand conditions. We propose D³PGSA, an enhanced reinforcement learning framework that integrates experience generation, feature-based DBSCAN clustering, and a dual-critic architecture to improve policy learning effectiveness.

The repository includes:
- The implementation of D³PGSA.
- Sample datasets and preprocessing scripts.
- Training, evaluation, and result reproduction instructions.
- Key experimental results.

## Repository Structure

```text
D3PGSA_EVCS_DynamicPricing/
├── README.md               # Project overview and instructions
│   
├── Code/
│   ├── D3PGSA.ipynb              # Main training and testing script
│   ├── chargenv.py             # D³PGSA model implementation
│   └── EV.py    # Python environment dependencies
├── Data/
│   ├── ACN-Data.csv               
│   └── EV Charging Reports.csv           
├── Results/
│   └── Figures/                
└── requirements.txt
    

# Experimental Results

This section presents key experimental results from our study, including training dynamics, comparative evaluations, and statistical analyses.

---

## 1. Training Dynamics

The following figure shows the training loss curve and policy evaluation during training.

### 1.1 Training Loss Curve
![Training Loss](Results/Figures/2.png)

### 1.2 Policy Evaluation Scores
![Policy Evaluation](Results/figures/evaluation_score.png)

---

## 2. Comparative Performance Across Algorithms

We evaluated D³PGSA against several baselines on multiple datasets.

### 2.1 Performance Comparison on LiteDemand Dataset
![LiteDemand Comparison](Results/figures/litedemand_bar.png)

### 2.2 Performance Comparison on HeavyHub Dataset
![HeavyHub Comparison](Results/figures/heavyhub_bar.png)

---

## 3. Quantitative Results

The following tables summarize the performance metrics across different algorithms.

### 3.1 Average Revenue (AR) Results

| Dataset        | GWO (AR)        | WSO (AR)        | DDPG (AR)       | TD3 (AR)        | DSAC (AR)       | D³PGSA (AR)     |
|----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| LiteDemand     | 788.37 ± 6.00    | 938.54 ± 46.05   | 870.46 ± 2.34    | 806.85 ± 5.12    | 1014.94 ± 1.53   | **1029.73 ± 0.98** |
| HeavyHub       | ...             | ...             | ...             | ...             | ...             | ...             |

### 3.2 Time-Weighted Reward Efficiency Index (TWREI) Results

| Dataset        | GWO (TWREI) | WSO (TWREI) | DDPG (TWREI) | TD3 (TWREI) | DSAC (TWREI) | D³PGSA (TWREI) |
|----------------|-------------|-------------|-------------|-------------|-------------|---------------|
| LiteDemand     | 0.03 ± 0.06  | 0.01 ± 0.02  | 0.05 ± 0.01  | 0.02 ± 0.01  | 0.01 ± 0.00  | **0.01 ± 0.00** |
| HeavyHub       | ...         | ...         | ...         | ...         | ...         | ...           |

---

## 4. Additional Visualizations

### 4.1 Scatter Plot of Test Results
![Test Scatter](Results/figures/test_scatter.png)

### 4.2 Histograms of Charging Demand Distribution
![Demand Histogram](Results/figures/demand_histogram.png)

---

