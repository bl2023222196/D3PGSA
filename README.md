# D³PGSA: A Dual-value Deep Deterministic Policy Gradient based on Sample Augmentation for Dynamic Pricing of EV Charging Station

This repository provides the code, dataset, and experimental results for our paper:

> **D³PGSA: A Dual-value Deep Deterministic Policy Gradient based on Sample Augmentation for Dynamic Pricing of EV Charging Stations**  
> Authors: [Anonymous for Review]  
> Status: Submitted to [Journal/Conference Name], under review.

## Overview

This project aims to address the dynamic pricing problem for electric vehicle charging stations (EVCS) under uncertain and dynamic demand conditions. We propose D³PGSA, an enhanced reinforcement learning framework that integrates experience generation model, feature generation model, feature-based DBSCAN clustering, and a dual-critic architecture to improve policy learning effectiveness.

The repository includes:
- The implementation of D³PGSA.
- Real datasets.
- Training, evaluation, and result reproduction instructions.
- Key experimental results.

## Repository Structure

```text
D3PGSA_EVCS_DynamicPricing/
├── README.md               # Project overview and instructions
├── Paper/
│   └── D3PGSA_Paper.pdf     # The submitted manuscript
├── Code/
│   ├── main.py              # Main training and testing script
│   ├── model.py             # D³PGSA model implementation
│   ├── utils.py             # Utility functions
│   └── requirements.txt     # Python environment dependencies
├── Data/
│   ├── sample_dataset.csv   # Example dataset (small-scale)
│   └── README.txt           # Instructions for obtaining full datasets
├── Results/
│   ├── figures/             # Experimental figures
│   └── logs/                # Training logs
└── scripts/
    └── run_experiment.sh    # Example script for running experiments
