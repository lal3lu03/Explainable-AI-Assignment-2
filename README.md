# Explainable AI Assignment - TabNet Model Explanations

This repository contains an Explainable AI assignment focused on interpreting a TabNet model trained on tabular weather data.

The project compares TabNet's built-in attention-based interpretability with general post-hoc explanation methods such as permutation feature importance, partial dependence plots, and LIME.

---

## Project overview

The goal of this assignment was to analyze how different explainability methods help understand a machine learning model's predictions.

The project uses the **Rain in Australia** dataset, which contains daily weather observations from multiple locations in Australia. The prediction task is to classify whether it will rain on the following day.

The model analyzed in this project is **TabNet**, an interpretable deep learning architecture for tabular data that uses sequential attention masks to select relevant features across decision steps.

---

## Explainability methods

The project compares four explanation approaches:

| Method                         | Explanation type          | Purpose                                                                     |
| ------------------------------ | ------------------------- | --------------------------------------------------------------------------- |
| Permutation Feature Importance | global, post-hoc          | Measures how prediction error changes when individual features are shuffled |
| Partial Dependence Plots       | global, post-hoc          | Shows how model predictions change as a feature value varies                |
| TabNet Attention Masks         | intrinsic, model-specific | Visualizes which features TabNet attends to during different decision steps |
| LIME                           | local, post-hoc           | Explains individual predictions using a local surrogate model               |

---

## Key analysis questions

This project investigates:

* which weather features are most important for predicting next-day rain
* whether TabNet's internal attention masks agree with model-agnostic explanation methods
* how global explanations differ from local instance-level explanations
* whether explanations align with domain intuition about humidity, pressure, wind, and rainfall
* how explainability methods can help debug misclassified examples

---

## Main findings

The analysis found that **Humidity3pm** was the strongest feature for predicting next-day rain. Other relevant features included **Pressure3pm**, **Rainfall**, **WindGustSpeed**, and wind-related measurements. TabNet's attention masks were broadly consistent with model-agnostic methods such as permutation importance and LIME, while LIME provided more detailed local explanations for individual predictions and misclassified samples.

Overall, the project showed that model-specific and model-agnostic explainability methods can complement each other when analyzing tabular deep learning models.

---

## Repository structure

```text
.
├── models/
├── src/
├── config.json
├── environment.yml
├── minimal_environment.yml
├── metrics.json
├── solution.ipynb
├── solution.html
├── train.py
└── README.md
```

---

## Files

| File or folder            | Purpose                                                       |
| ------------------------- | ------------------------------------------------------------- |
| `solution.ipynb`          | main executed notebook with analysis, plots, and explanations |
| `solution.html`           | exported HTML version of the notebook                         |
| `train.py`                | model training script                                         |
| `models/`                 | saved model artifacts                                         |
| `src/`                    | supporting source code                                        |
| `config.json`             | model and experiment configuration                            |
| `metrics.json`            | saved model metrics                                           |
| `environment.yml`         | full conda environment                                        |
| `minimal_environment.yml` | smaller fallback environment                                  |

---

## Environment setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate xai_model_explanation
```

If the full environment causes issues, use the minimal environment:

```bash
conda env create -f minimal_environment.yml
conda activate xai_model_explanation
```

Then start Jupyter Lab:

```bash
jupyter lab
```

Open:

```text
solution.ipynb
```

---

## Model

The project uses **TabNet**, a neural architecture designed for tabular data.

TabNet uses sequential attention to select features during multiple decision steps. This makes it possible to inspect feature usage through attention masks and compare these internal explanations with external explainability methods.

---

## Dataset

The project uses the **Rain in Australia** dataset from Kaggle.

The dataset contains daily weather observations from Australian weather stations. The classification target is whether it rains on the following day.

---

## Portfolio context

This repository is an academic Explainable AI project. It is included in my portfolio because explainability is especially relevant for trustworthy AI systems, biomedical AI, and scientific machine learning workflows.

It demonstrates experience with:

* explainable AI methods
* tabular deep learning
* model interpretation
* feature importance analysis
* local and global explanations
* Jupyter-based scientific reporting

For my main biomedical AI and protein modeling work, see:

* `https://github.com/lal3lu03/PockNet`
* `https://github.com/lal3lu03/MMCLFMI`

---

## Maintainer

Maximilian Hageneder 
LinkedIn: `https://www.linkedin.com/in/maximilian-hageneder-ai`
