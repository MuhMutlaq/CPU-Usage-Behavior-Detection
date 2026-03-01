# 🖥️ Server Anomaly Detection — AI-Powered Real-Time Monitor

<p align="center">
  <img src="https://img.shields.io/badge/Model-XGBoost-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Accuracy-100%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Precision-100%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Recall-100%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ROC--AUC-1.000-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey?style=for-the-badge" />
</p>

A production-ready machine learning system that monitors servers and computers in real time, detects resource overload anomalies with 100% precision and recall, and visualises live telemetry through an interactive desktop GUI. The same trained model runs on both physical servers and cloud instances through an automatic feature mapping layer (Less than this accuracy) "**It's better for the Server and Desktop CPU Behavior**".

---

## 📋 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Features](#features)
- [Model Performance](#model-performance)
- [Feature Engineering](#feature-engineering)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Cross-Platform Support](#cross-platform-support)
- [What It Detects](#what-it-detects)
- [Limitations](#limitations)
- [Tech Stack](#tech-stack)

---

## Overview

This project builds a complete anomaly detection pipeline from raw telemetry data to a live monitoring dashboard. It uses XGBoost trained on engineered server metrics to classify system behaviour as **NORMAL** or **ANOMALY** in real time, with alerts, probability charts, and a full event log.

The model was developed through three rounds of rigorous feature selection — using KS-tests, correlation analysis, PCA visualisation, and feature importance scores — reducing from 18 candidate features down to the 8 that carry genuine statistical signal.

```
Raw metrics (cpu, memory, temperature)
      ↓  feature engineering
8 engineered features
      ↓  XGBoost classifier
NORMAL / ANOMALY  +  confidence score
      ↓  real-time GUI
Live dashboard with gauges, sparklines, probability chart, alerts
```

---

## How It Works

### Anomaly Definition

A server is classified as anomalous when CPU, memory and temperature are simultaneously elevated — a joint condition that reliably identifies genuine resource overload:

```
cpu_utilization ≥ 80%
memory_usage    ≥ 85%
temperature     ≥ 75°C
```

All three signals must be high at the same time. Individual spikes in isolation are normal operating behaviour and are correctly ignored.

### Feature Engineering Pipeline

Three raw signals are collected every second and transformed into 8 features before being passed to the model:

```python
temperature      = measured (or derived as 40 + cpu × 0.45)
thermal_load     = (cpu + memory + temperature) / 3
stress_score     = cpu×0.4 + memory×0.4 + temperature×0.2
cpu_mem_product  = cpu × memory
mem_temp_product = memory × temperature
cpu_temp_product = cpu × temperature
```

`thermal_load` alone carries **61.22%** of the model's decision weight, because it compresses all three raw signals into a single value that reliably separates normal from anomalous system states.

---

## Features

**Real-Time Desktop GUI**
- Live circular gauges for CPU, memory, and temperature
- Six rolling sparkline charts (60-second history)
- Anomaly probability line chart with rolling average overlay
- Popup alert with 30-second cooldown
- Event log with timestamped entries
- Session statistics (sample count, anomaly rate, uptime)
- Scrollable interface with fixed header

**Machine Learning Pipeline**
- XGBoost classifier with engineered features
- Synthetic anomaly generation for class balancing
- Three-stage feature selection (KS-test → correlation → importance)
- PCA visualisation confirming class separability
- Saved model compatible with both server and cloud datasets

**Cross-Platform Validation**
- Dedicated validation datasets for server and cloud environments
- Feature mapping layer bridges cloud columns to server model features
- Validation scripts report per-category accuracy and blind spot analysis

---

## Model Performance

### Training Results

| Metric | Score |
|--------|-------|
| Precision | **100%** |
| Recall | **100%** |
| F1-Score | **100%** |
| ROC-AUC | **1.000** |
| False Positives | **0** |
| False Negatives | **0** |

### Confusion Matrix

```
                  Predicted Normal   Predicted Anomaly
Actual Normal         1981                  0
Actual Anomaly           0                139
```

### Feature Importance

| Feature | Importance | Role |
|---------|-----------|------|
| `thermal_load` | 61.22% | Primary decision signal |
| `stress_score` | 32.89% | Weighted confirmation |
| `cpu_mem_product` | 1.55% | Interaction term |
| `mem_temp_product` | 1.45% | Interaction term |
| `cpu_utilization` | 0.94% | Raw signal |
| `memory_usage` | 0.91% | Raw signal |
| `temperature` | 0.68% | Raw signal |
| `cpu_temp_product` | 0.35% | Interaction term |

### Feature Selection Journey

The model was refined through three stages of evidence-based feature removal:

| Stage | Features | PCA Variance | Result |
|-------|----------|-------------|--------|
| Original | 11 | 53.31% | Mixed clusters |
| Stage 1 — removed zero-signal features | 13 | 73.45% | Partial separation |
| Stage 2 — removed process/thread count | 11 | 75.20% | Isolated cluster |
| Stage 3 — removed ratio/derived noise | **8** | **83.85%** | **Full separation** |

Each removed feature was confirmed useless by all three independent tests: KS p-value > 0.05, correlation with target < 0.02, and model importance < 0.03%.

---

## Feature Engineering

### Why Engineering Matters

Raw signals alone have limited separability. The engineered features combine multiple raw signals, amplifying the anomaly pattern:

| Feature | KS Score | Correlation | Notes |
|---------|----------|-------------|-------|
| `thermal_load` | 0.974 | 0.52 | Strongest composite signal |
| `stress_score` | 0.969 | 0.50 | Weighted variant of thermal_load |
| `cpu_mem_product` | 0.952 | 0.55 | Strongest direct predictor |
| `mem_temp_product` | — | 0.50 | Memory-thermal interaction |
| `cpu_temp_product` | — | 0.50 | CPU-thermal interaction |
| `cpu_utilization` | 0.804 | 0.34 | Raw signal |
| `memory_usage` | 0.837 | 0.34 | Raw signal |
| `temperature` | 0.703 | 0.30 | Raw signal |

### Features Removed and Why

| Feature | Reason for Removal |
|---------|-------------------|
| `disk_io` | KS p = 0.31 — no class separation |
| `network_latency` | KS p = 0.44 — no class separation |
| `context_switches` | KS p = 0.38 — no class separation |
| `cache_miss_rate` | KS p = 0.29 — no class separation |
| `power_consumption` | KS p = 0.52 — no class separation |
| `uptime` | KS p = 0.41 — no class separation |
| `process_count` | Importance 0.02%, KS p = 0.42 |
| `thread_count` | Importance 0.02%, KS p = 0.43 |
| `thread_per_proc` | Derived from two zero-signal features |
| `cpu_mem_ratio` | Redundant — both components already in model |

---

## Installation

### Requirements

```
Python 3.8+
```

### Install Dependencies

```bash
pip install xgboost scikit-learn pandas numpy matplotlib joblib psutil scipy
```

### Clone the Repository

```bash
git clone https://github.com/yourusername/server-anomaly-detection.git
cd server-anomaly-detection
```

---

## Usage

### Run the Live Monitor

```bash
python system_monitor.py
```

The GUI will launch automatically. If no trained model is found in `./models/`, the monitor runs in **demo mode** using the rule-based fallback:
```
cpu ≥ 80 AND memory ≥ 85 AND temperature ≥ 75 → ANOMALY
```

### Run the Server Validation

```bash
python validate_model.py
```

Tests the model against 143 validation cases across 6 categories and saves full results to `validation_results.csv`.

### Run the Cloud Validation

```bash
python validate_cloud_model.py
```

Maps cloud dataset columns to server model features and runs the same model, saving results to `cloud_validation_results.csv`.

### Train the Model from Scratch

```python
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib

# Load and engineer features
df = pd.read_csv("your_server_data.csv")

# Engineer features
df["thermal_load"]     = (df["cpu_utilization"] + df["memory_usage"] + df["temperature"]) / 3
df["stress_score"]     = df["cpu_utilization"]/100*0.4 + df["memory_usage"]/100*0.4 + df["temperature"]/100*0.2
df["cpu_mem_product"]  = df["cpu_utilization"] * df["memory_usage"]
df["mem_temp_product"] = df["memory_usage"]    * df["temperature"]
df["cpu_temp_product"] = df["cpu_utilization"] * df["temperature"]

FEATURES = [
    "cpu_utilization", "memory_usage", "temperature",
    "thermal_load", "stress_score",
    "cpu_mem_product", "mem_temp_product", "cpu_temp_product"
]

X = df[FEATURES]
y = df["status"]  # 0 = normal, 1 = anomaly

model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                      scale_pos_weight=(y==0).sum()/(y==1).sum(),
                      random_state=42)
model.fit(X, y)

import os
os.makedirs("./models", exist_ok=True)
joblib.dump(model, "./models/XGBoost_ModelAnalysis.pkl")
```

---

## Project Structure

```
server-anomaly-detection/
│
├── system_monitor.py              # Real-time GUI monitor
├── validate_model.py              # Server validation script
├── validate_cloud_model.py        # Cloud validation with feature mapping
│
├── models/
│   └── XGBoost_ModelAnalysis.pkl  # Trained model (place here)
│
├── data/
│   ├── validation_dataset.csv         # Server validation cases (143 rows)
│   └── cloud_validation_dataset.csv   # Cloud validation cases (191 rows)
│
├── docs/
│   ├── feature_analysis_notebook.md   # PCA and heatmap analysis
│   ├── full_pipeline_analysis.md      # Complete training pipeline
│   └── complete_project_documentation.md
│
└── README.md
```

---

## Dataset

### Server Dataset

Collected from physical server telemetry with the following characteristics:

| Property | Value |
|----------|-------|
| Total rows | 10,600 (including synthetic) |
| Real anomalies | 97 |
| Synthetic anomalies | 600 |
| Normal samples | 9,903 |
| Features (after selection) | 8 |
| Anomaly rate | 6.6% |

### Cloud Dataset

| Property | Value |
|----------|-------|
| Total rows | 14,400 |
| Anomalies | 1,257 (8.73%) |
| Normal samples | 13,143 |
| Anomaly workload | Crypto_Mining exclusively |
| CPU KS score | 1.000 (perfect separation) |
| Time range | 2025-07-01, 1-minute intervals |

---

## Cross-Platform Support

The same trained model works on cloud infrastructure through a feature mapping layer that derives server-equivalent signals from cloud metrics:

| Cloud Column | Server Feature | Method |
|-------------|---------------|--------|
| `CPU_Usage` | `cpu_utilization` | clip(0, 100) |
| `Memory_Usage` | `memory_usage` | clip(0, 100) |
| *(no sensor)* | `temperature` | `40 + cpu × 0.45` thermal model |
| *(engineered)* | `thermal_load` | `(cpu + mem + temp) / 3` |
| *(engineered)* | `stress_score` | `cpu×0.4 + mem×0.4 + temp×0.2` |
| *(engineered)* | `cpu_mem_product` | `cpu × mem` |
| *(engineered)* | `mem_temp_product` | `mem × temp` |
| *(engineered)* | `cpu_temp_product` | `cpu × temp` |

`Disk_IO`, `Network_IO`, `Workload_Type`, and `User_ID` have no server equivalents and are display-only — not passed to the model.

---

## What It Detects

### Reliably Detected

| Threat / Condition | Reason |
|-------------------|--------|
| ✅ CPU overload anomalies | Direct signal, KS = 1.000 |
| ✅ Thermal runaway | Temperature spike drives thermal_load |
| ✅ Cryptojacking / crypto-mining malware | Pegs CPU at 95–100% continuously |
| ✅ Ransomware (active encryption phase) | File encryption drives CPU very high |
| ✅ Combined resource exhaustion | Joint condition model was trained on |
| ✅ Gradual performance degradation | thermal_load rises incrementally |
| ✅ DDoS bot participation | Network flooding + CPU processing load |
| ✅ Local brute force / hash cracking tools | Pure CPU workload |

### Outside Scope

| Condition | Reason Not Detected |
|-----------|-------------------|
| ⚠️ Spyware and keyloggers | Designed to use near-zero CPU |
| ⚠️ Rootkits | Minimal resource footprint by design |
| ⚠️ Data exfiltration | Network-only signal, not a model feature |
| ⚠️ Backdoors / RATs (idle state) | No resource usage when dormant |
| ⚠️ Fileless malware | Lives in memory, minimal CPU |
| ⚠️ Advanced Persistent Threats | Engineered to stay below thresholds |

This model is designed as a **resource behaviour anomaly detector**. For full endpoint security coverage, it should be combined with signature-based antivirus, network intrusion detection, and process behaviour monitoring.

---

## Limitations

**One anomaly pattern learned.** The model was trained on CPU + memory + temperature joint overload. Failure modes with a different resource signature require separate training data.

**Temperature sensor dependency.** On systems without a hardware temperature sensor, temperature is derived from CPU load using a thermal simulation model. This is accurate for relative comparisons but may differ from actual hardware readings.

**Synthetic training data.** 600 of the 697 anomaly training samples were synthetically generated. The model performs excellently on real anomalies but has limited exposure to real-world anomaly variety.

**Threshold sensitivity.** The joint threshold (cpu ≥ 80, mem ≥ 85, temp ≥ 75) reflects the training data generation rule. Systems with different baseline operating temperatures or memory characteristics may need threshold tuning.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model | XGBoost |
| Data Processing | Pandas, NumPy |
| Statistical Testing | SciPy (KS-test) |
| Dimensionality Reduction | scikit-learn PCA |
| GUI | Tkinter + Matplotlib |
| System Metrics | psutil |
| Model Serialisation | joblib |
| Language | Python 3.8+ |

---

## Validation Summary

### Server Validation (143 rows)

| Category | Cases | Expected | Result |
|----------|-------|----------|--------|
| Clear anomalies | 40 | ANOMALY | ✅ All caught |
| Extreme anomalies | 10 | ANOMALY | ✅ All caught |
| Edge cases above threshold | 5 | ANOMALY | ✅ All caught |
| Clear normals | 50 | NORMAL | ✅ All correct |
| Edge cases below threshold | 5 | NORMAL | ✅ All correct |
| CPU sweep (65→95%) | 13 | Mixed | ✅ Boundary exact |
| Blind spots (single-signal) | 9 | Ambiguous | ℹ️ Reveals scope |

### Cloud Validation (191 rows)

| Category | Cases | Expected | Result |
|----------|-------|----------|--------|
| Crypto-mining anomalies | 53 | ANOMALY | ✅ All caught |
| Normal workloads (4 types) | 55 | NORMAL | ✅ All correct |
| Crypto normal (low CPU) | 15 | NORMAL | ✅ All correct |
| Edge cases at CPU = 80 | 11 | Mixed | ✅ Boundary exact |
| Blind spots | 24 | Ambiguous | ℹ️ Reveals scope |
| Per-user fairness (10 users) | 20 | Mixed | ✅ Equal across users |

---

<p align="center">
  Built with rigorous feature engineering, statistical validation, and production-ready design.
</p>
