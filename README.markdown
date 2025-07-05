# 🧬 Majorana Zero Modes Detection via Quantum Simulation and Machine Learning

This project implements two complementary pipelines for detecting **Majorana Zero Modes (MZMs)** in 1D Kitaev chains using quantum simulations and machine learning. MZMs are exotic zero-energy states that appear in topological superconductors and hold promise for quantum computing.

We simulate both:
- **Wavefunction-based localization pipeline**  
- **Transport-based conductance map pipeline**

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Method 1: Wavefunction-Based Localization](#-method-1-wavefunction-based-localization)
- [Method 2: Conductance Map-Based Detection](#-method-2-conductance-map-based-detection)
- [Machine Learning Models](#-machine-learning-models)
- [Results & Visualizations](#-results--visualizations)
- [Folder Structure](#-folder-structure)
- [References](#-references)
- [Contributions & Citation](#-contributions--citation)

---

## 📖 Overview

We explore Majorana detection through two physical representations:

| Approach                | Input                     | Output                        | Signature |
|-------------------------|---------------------------|-------------------------------|-----------|
| **Localization-Based**  | Hamiltonian parameters    | BdG spectrum + edge-localization | Zero-energy mode + ξ |
| **Conductance-Based**   | Quantum transport setup   | Conductance maps (dI/dV)      | Zero-bias peak (ZBP) |

Each pipeline generates data, trains ML models, and analyzes diagnostics (PCA, histograms, phase diagrams).

---

## 🧪 Method 1: Wavefunction-Based Localization

This method constructs the **Bogoliubov–de Gennes (BdG)** Hamiltonian from Kitaev chain parameters and analyzes its eigenstructure to detect localized zero modes.

### 🔧 Simulation

- Model: 1D Kitaev chain
- Parameters swept:
  - `µ`: chemical potential
  - `t`: hopping term
  - `∆`: pairing potential
  - `σ`: disorder strength
  - `L`: chain length

### 🔬 Output

- Eigenvalues (`Eₙ`)
- Eigenvectors (`ψₙ`)
- **Localization length (ξ)** via exponential fit:  
  `|ψ_j|² = |u_j|² + |v_j|² ∼ exp(−2j/ξ)`

### 🧠 ML Model

- Input: `[L, µ, t, ∆, σ]`
- Output: `[ℓ, λ₁, ..., λ₁₈₀]`  
  - ℓ: edge-localization score  
  - λₖ: BdG eigenvalues

Architecture: MLP with dual heads (edge-localization + spectrum regression)

---

## ⚡ Method 2: Conductance Map-Based Detection

This method simulates a **semiconductor-superconductor nanowire** and measures conductance across varying gate voltages and source-drain bias.

### 🔧 Setup

- Simulated in Kwant
- Parameters swept:
  - `µ`: gate-controlled chemical potential
  - `Vbias`: source-drain voltage
  - Fixed `∆`, `t`, `disorder`, and confinement

### 🔬 Output

- **Conductance map** (2D array of dI/dV values)
- Grid shape: 30×30 over (`µ`, `Vbias`)
- 4,000+ maps simulated

### 🧠 ML Model

- Input: 2D conductance maps  
- Label: topological vs trivial (based on ZBP at `Vbias ≈ 0`)  
- Classifier: CNN / XGBoost / engineered features

---

## 📈 Results & Visualizations

### 🧊 Wavefunction-Based

#### 🔹 Edge Localization Histogram
Shows MZMs appear at edges (score ≈ 1) while bulk states cluster near 0.

![Edge Localization Histogram](assets/edge_localization_histogram.png)

#### 🔹 PCA of Engineered Features
Topological and trivial phases separate clearly in reduced feature space.

![PCA of Engineered Features](assets/pca_engineered_features.png)

#### 🔹 Energy Spectrum with MZMs
Two zero-energy modes indicate Majorana formation.

![Energy Spectrum](assets/energy_spectrum_mzm.png)

---

### ⚡ Conductance-Based

#### 🔹 Conductance Maps with ZBPs
Topological: Clear zero-bias peak  
Trivial: No peak or split peaks

![Our Conductance Map](assets/our_conductance_map.png)

#### 🔹 Phase Diagram (µ vs ∆)
Regions with high frequency of MZMs under specific gate control.

![Phase Diagram](assets/phase_diagram_mu_delta.png)

---

## 🧠 Machine Learning Summary

| Pipeline       | Input                | Output                | ML Model        |
|----------------|----------------------|------------------------|-----------------|
| Wavefunction   | `[L, µ, t, ∆, σ]`     | `ℓ`, `λ₁–₁₈₀`         | Enhanced MLP    |
| Conductance    | 2D (30×30) maps       | `0` (trivial) or `1` (MZM) | XGBoost or CNN |

---



