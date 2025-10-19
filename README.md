# FedAT: Adaptive Federated Learning with Time-Varying Computing Resources

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-brightgreen.svg)]()
[![Framework](https://img.shields.io/badge/PyTorch-2.0+-red.svg)]()
[![Paper](https://img.shields.io/badge/Preprint-ICTExpress%202025-blue.svg)]()

This repository provides the official implementation of **FedAT**, an adaptive federated learning framework designed to achieve **faster convergence** and **fairer client participation** under **heterogeneous and time-varying computing resources**.

---

## 🔍 Overview

Federated learning (FL) enables distributed model training without data centralization. However, in realistic scenarios, participating devices exhibit **high heterogeneity** in computation and communication capabilities, leading to **unbalanced training** and **slow convergence**.

To address these issues, **FedAT** introduces an *adaptive co-adjustment* mechanism that dynamically adapts the **local epoch number** and **learning rate** according to predicted device iteration time. Combined with a **hybrid client selection** and **weighted aggregation**, FedAT achieves both **faster convergence** and **fairer resource utilization**.

---

## 🧩 Key Contributions

- ⚙️ **Adaptive Local Training**  
  Dynamically adjusts local epochs and learning rates based on predicted device speed and training stability.

- ⏱️ **AR(1) Iteration Time Predictor**  
  Uses a first-order autoregressive model to estimate per-round computation time and mitigate temporal fluctuation.

- 🔀 **Hybrid Client Selection Strategy**  
  Combines probabilistic sampling with directed selection to balance participation between fast and slow devices.

- 🧮 **Weighted Aggregation Rule**  
  Integrates both local epoch counts and data sizes for more representative global updates.

- 📊 **Fairness-Aware Evaluation**  
  Introduces *participation variance* metric to quantify the fairness of client involvement.

---

## 📁 Repository Structure


---

## ⚙️ Installation

### Option A: Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate fedat



python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
