# D³PGSA: A Dual-value Deep Deterministic Policy Gradient based on Sample Augmentation for Dynamic Pricing of EV Charging Station

This repository provides the code, dataset, and experimental results for our paper:

> **D³PGSA: A Dual-value Deep Deterministic Policy Gradient based on Sample Augmentation for Dynamic Pricing of EV Charging Station**  
> Status: Submitted to IEEE Transactions on Intelligent Transportation Systems, In Revision.

---

## Overview

This project addresses the dynamic pricing problem for electric vehicle charging stations (EVCS) under uncertain and dynamic demand conditions.  
We propose D³PGSA, an enhanced reinforcement learning framework that integrates experience generation, feature-based DBSCAN clustering, and a dual-critic architecture to improve policy learning effectiveness.

The repository includes:
- The implementation of D³PGSA.
- Datasets.
- Training, evaluation, and result reproduction instructions.
- Key experimental results.

---

## Repository Structure

```text
D3PGSA/
├── README.md               # Project overview and instructions
│   
├── Code/
│   ├── D3PGSA.ipynb             
│   ├── chargenv.py             
│   ├── EV.py                 
│   └── Comparison_Algorithm/
├── datasets/
│   ├── ACN-Data.csv               
│   └── EV Charging Reports.csv           
├── Results/
├── Images/
└── requirements.txt
```
---


## 🔧 Running the Code

### 1. Environment Setup

Install all required Python packages using:

```bash
pip install -r requirements.txt
```
---
### 2. Data Preparation

Please ensure your dataset is stored at:

```bash
./datasets/{dataset_name}.csv
```
---

### 3. Start Training

```bash
python D3PGSA.ipynb
```
The training includes:

Actor-Critic model with dual Q-networks

Experience generation and clustering using simsiam + DBSCAN

Adaptive exploration via noise decay

Soft updates and performance monitoring

---
### 4. Key Configurations
Modify the following variables in the script as needed:
| Parameter        | Description                          | Example      |
|------------------|--------------------------------------|--------------|
| `Num_data`       | Number of each dataset               | `5`          |
| `num_episodes`   | Total training episodes              | `1000`       |
| `action_bound`   | Max charging power per time slot     | `0.5`        |
| `buffer_size`    | Size of replay buffer                | `40000`      |
| `sigma`          | Initial exploration noise (Gaussian) | `0.5`        |
| `critic_lr`      | Learning rate of critic              | `0.002`      |
| `actor_lr`       | Learning rate of actor               | `0.002`      |
| `gamma`       | Decay factor               | `0.98`      |
| `tau`       | Soft update parameter               | `0.001`      |
| `batch_size`       | Experience sample size               | `128`      |
### 5. Outputs
After training, the following files are generated:
| Output Type    | Path                                                | Description                        |
|----------------|-----------------------------------------------------|------------------------------------|
| Trained Actor  | `./model/actor_D3PGSA{}.pth`                       | Best actor model checkpoint        |
| Training Log   | `./result/train/D3PGSA{}.csv`                      | Episode returns saved as `.csv`   |
| Training Curve | Shown via `matplotlib.pyplot`                      | Return vs. episode plot            |
### 6. Evaluate Trained Model
```bash
agent.actor.load_state_dict(torch.load('./model/actor_D3PGSA{}.pth'))

```
Then interact with the environment using env.step().

# Experimental Results

This section presents key experimental results from our study, including training dynamics, comparative evaluations, and statistical analyses.

---

## 1. Training Dynamics

The learning behavior of D³PGSA during training is illustrated below.

### 1.1 Training Reward Curves

![Demand Histogram](Images/8.png)

*Figure 1: Reward curve of D³PGSA and other algorithms over epochs.*

---

  ### 1.2  Solution Time Curves 

![Demand Histogram](Images/7.png)

*Figure 2: Reward curve of D³PGSA and other algorithms over epochs.*

---

## 2. Comparative Performance Across Algorithms

Performance comparisons between D³PGSA and baseline algorithms on different datasets are presented.

### 2.1 LiteDemand Dataset

![Demand Histogram](Images/7.png)

*Figure 3: Average revenue achieved by different algorithms on the LiteDemand dataset. D³PGSA outperforms all baselines.*

---

### 2.2 HeavyHub Dataset
![Demand Histogram](Images/7.png)

*Figure 4: Average revenue comparison on the HeavyHub dataset under high-demand conditions.*

---

## 3. Quantitative Results

The following tables summarize detailed performance metrics across algorithms.

### 3.1 Average Revenue (AR)

| Dataset        | GWO (AR)        | WSO (AR)        | DDPG (AR)       | TD3 (AR)        | DSAC (AR)       | D³PGSA (AR)         |
|----------------|-----------------|-----------------|-----------------|-----------------|-----------------|---------------------|
| LiteDemand     | 788.37 ± 6.00    | 938.54 ± 46.05   | 870.46 ± 2.34    | 806.85 ± 5.12    | 1014.94 ± 1.53   | **1029.73 ± 0.98**   |
| HeavyHub       | ...             | ...             | ...             | ...             | ...             | ...                 |

*Table 1: Average Revenue (AR) results across datasets. Bold values indicate the best-performing method.*

---

### 3.2 Time-Weighted Reward Efficiency Index (TWREI)

| Dataset        | GWO (TWREI) | WSO (TWREI) | DDPG (TWREI) | TD3 (TWREI) | DSAC (TWREI) | D³PGSA (TWREI)     |
|----------------|-------------|-------------|-------------|-------------|-------------|--------------------|
| LiteDemand     | 0.03 ± 0.06  | 0.01 ± 0.02  | 0.05 ± 0.01  | 0.02 ± 0.01  | 0.01 ± 0.00  | **0.01 ± 0.00**    |
| HeavyHub       | ...         | ...         | ...         | ...         | ...         | ...                |

*Table 2: Time-Weighted Reward Efficiency Index (TWREI) results. Lower values indicate better efficiency.*

---

## 4. Additional Visualizations

Complementary visualizations provide further insights into the experimental outcomes.

### 4.1 Scatter Plot of Test Results

![Demand Histogram](Images/7.png)

*Figure 5: Scatter plot showing the distribution of revenues across various test scenarios.*

---

### 4.2 Charging Demand Distribution

![Demand Histogram](Images/7.png)

*Figure 6: Histogram of charging demand distribution, illustrating peak and valley periods.*

---
