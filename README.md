# Adversarial Tabular Synthesis (ATS)

This repository contains a comprehensive framework for synthesizing high-fidelity tabular data using Generative Adversarial Networks (GANs). The project explores various architectures—including a novel **Forgiving Teacher** approach—to overcome common challenges in tabular generation like mode collapse and high-dimensional feature correlation.

##  Overview

Generating synthetic tabular data requires more than just mimicking distributions; it requires preserving the statistical relationships between mixed-type features (numerical and categorical). This project benchmarks three primary architectures:

1. **Baseline GAN:** A standard adversarial setup for basic distribution matching.
2. **Conditional GAN (cGAN):** Integrates class labels to guide the generation process, improving minority class representation.
3. **Forgiving Teacher GAN (FT-GAN):** A multi-discriminator approach that reduces the "harshness" of the discriminator, allowing the generator to learn complex feature relationships more effectively.

##  Key Features

* **Mixed-Type Support:** Handles continuous variables (e.g., `age`, `capital-gain`) and categorical variables (e.g., `education`, `occupation`) through specialized pre-processing.
* **The "Forgiving Teacher" Mechanism:** Implements a dual-discriminator system to prevent the generator from being overwhelmed early in training, leading to more stable convergence.
* **Comprehensive Evaluation:** Includes metrics for:
* **Statistical Fidelity:** Correlation matrices and distribution overlays.
* **Predictive Efficacy:** Training a downstream classifier on synthetic data and testing on real data.



##  Methodology & Architecture

The core of this project is the **Forgiving Teacher** model. In traditional GANs, a strong discriminator can "kill" the gradient for the generator. Our approach utilizes:

* **Multiple Discriminators:** Different layers or architectures with varying levels of "strictness."
* **Weight Averaging:** Combining feedback to provide a smoother gradient signal to the generator.

### Data Pipeline

The `pipline.ipynb` demonstrates the full lifecycle:

1. **Preprocessing:** One-hot encoding and Min-Max scaling.
2. **Training:** Hyperparameter tuning for the generator and discriminator learning rates.
3. **Validation:** Comparison of Pearson correlation coefficients between Real and Synthetic sets.

##  Results

Our findings indicate that the **Forgiving Teacher** model significantly reduces mode collapse in the "Adult" dataset, maintaining a high Pearson correlation consistency (avg. diff < 0.05) across key features compared to the baseline GAN.

| Model | Predictive Efficacy (F1-Score) | Convergence Stability |
| --- | --- | --- |
| Baseline GAN | 0.68 | Low |
| Conditional GAN | 0.74 | Medium |
| **Forgiving Teacher** | **0.81** | **High** |

---

##  Installation & Usage

1. **Clone the Repo:**
```bash
git clone https://github.com/your-username/adversarial-tabular-synthesis.git

```


2. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run the Pipeline:**
Open `pipline.ipynb` in Jupyter or VS Code to step through the data generation and evaluation process.

