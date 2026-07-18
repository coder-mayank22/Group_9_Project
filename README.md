<h1 align="center">Fake Review Detection and Trust Score Engine for E-commerce Platforms</h1>

<p align="center">
An AI-powered machine learning system for detecting fake product reviews and computing product Trust Scores using Natural Language Processing and Behavioral Feature Engineering.
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white">
<img src="https://img.shields.io/badge/Machine%20Learning-XGBoost-success">
<img src="https://img.shields.io/badge/NLP-TF--IDF-orange">
<img src="https://img.shields.io/badge/Dataset-Amazon%20Labeled%20Fake%20Reviews-red">
<img src="https://img.shields.io/badge/License-Academic-lightgrey">
</p>

---

# Table of Contents

- Overview
- Features
- Tech Stack
- Dataset
- Methodology
- Workflow
- Machine Learning Models
- Results
- Trust Score Engine
- Project Structure
- Installation
- Usage
- Future Scope
- Team

---

# Overview

Online product reviews significantly influence purchasing decisions. However, fake and bot-generated reviews manipulate product ratings, mislead customers, and reduce trust in e-commerce platforms.

This project presents a complete **Fake Review Detection and Trust Score Engine** that combines **Natural Language Processing (NLP)** with **behavioral metadata** to accurately classify reviews as genuine or fake. The system further computes a **Trust Score** for each product by aggregating the authenticity of its reviews.

---

# Features

- Fake review detection using Machine Learning
- TF-IDF based Natural Language Processing
- Behavioral feature engineering
- Product-level Trust Score generation
- Comparison of multiple ML algorithms
- Five-fold Cross Validation
- High accuracy using XGBoost
- Easily extendable for deployment

---

# Tech Stack

<p align="left">

<a href="https://www.python.org">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="48"/>
</a>

<a href="https://numpy.org">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="48"/>
</a>

<a href="https://pandas.pydata.org">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" width="48"/>
</a>

<a href="https://scikit-learn.org">
<img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" width="48"/>
</a>

<a href="https://xgboost.ai">
<img src="https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/xgboost.png" width="48"/>
</a>

<a href="https://matplotlib.org">
<img src="https://upload.wikimedia.org/wikipedia/commons/8/84/Matplotlib_icon.svg" width="48"/>
</a>

<a href="https://seaborn.pydata.org">
<img src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" height="48"/>
</a>

<a href="https://textblob.readthedocs.io">
<img src="https://raw.githubusercontent.com/sloria/TextBlob/master/docs/_static/textblob-logo.png" height="48"/>
</a>

<a href="https://colab.research.google.com">
<img src="https://colab.research.google.com/img/colab_favicon_256px.png" width="48"/>
</a>

</p>

---

# Dataset

**Dataset:** Amazon Labeled Fake Reviews

## Dataset Statistics

| Category | Count |
|-----------|------:|
| Total Reviews | 50,000 |
| Genuine Reviews | 25,281 |
| Fake Reviews | 24,719 |

Each review contains:

- Rating
- Review Title
- Review Text
- Images
- Product ID
- User ID
- Timestamp
- Helpful Votes
- Verified Purchase
- User Timestamp
- User Review Burst
- Label (Fake/Genuine)

---

# Methodology

The proposed pipeline consists of five major stages.

```
Data Collection
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Model Training
        │
        ▼
Cross Validation
        │
        ▼
Evaluation
        │
        ▼
Trust Score Generation
```

---

# Data Preprocessing

The following preprocessing techniques were applied:

- Lowercase conversion
- HTML tag removal
- Punctuation removal
- Digit removal
- Whitespace normalization
- Missing value handling
- Title and review text concatenation

---

# Feature Engineering

## Text Features

- TF-IDF Vectorization
- Unigrams
- Bigrams

## Behavioral Features

- Rating
- Helpful Votes
- Verified Purchase
- User Timestamp
- User Review Burst
- Word Count
- Text Length
- Capitalization Ratio
- Exclamation Count
- Image Availability

---

# Machine Learning Models

Three supervised learning algorithms were trained and compared.

| Algorithm | Purpose |
|------------|----------|
| Logistic Regression | Baseline Linear Classifier |
| Random Forest | Ensemble Learning |
| XGBoost | Gradient Boosted Decision Trees |

---

# Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

---

# Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---------|----------|-----------|--------|----------|
| Logistic Regression | 79.68% | 86.14% | 70.55% | 77.57% |
| Random Forest | 90.70% | 98.94% | 82.20% | 89.80% |
| **XGBoost** | **97.95%** | **98.97%** | **96.86%** | **97.90%** |

The experimental results demonstrate that **XGBoost** significantly outperforms the other classifiers by effectively learning complex interactions between textual and behavioral features.

---

# Trust Score Engine

The Trust Score Engine computes a credibility score for each product using the fake-review probabilities generated by the XGBoost model.

```
Trust Score = 1 − Average(Fake Review Probability)
```

Interpretation:

| Trust Score | Meaning |
|--------------|----------|
| 0.90 – 1.00 | Highly Trustworthy |
| 0.75 – 0.89 | Mostly Genuine |
| 0.50 – 0.74 | Moderate Risk |
| Below 0.50 | Suspicious Product |

---

# Workflow

```
Amazon Reviews
       │
       ▼
Cleaning
       │
       ▼
Feature Extraction
       │
       ▼
TF-IDF + Behavioral Features
       │
       ▼
Model Training
       │
       ▼
Fake Review Prediction
       │
       ▼
Trust Score Generation
```

---

# Project Structure

```
Fake-Review-Detection/
│
├── Dataset/
│   └── final_labeled_fake_reviews.csv
│
├── Models/
│   ├── XGBoost.pkl
│   ├── RandomForest.pkl
│   └── LogisticRegression.pkl
│
├── Notebook/
│   └── FakeReviewDetection.ipynb
│
├── Images/
│
├── README.md
│
└── requirements.txt
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/your-username/Fake-Review-Detection.git
```

Move into the project directory

```bash
cd Fake-Review-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the notebook

```bash
jupyter notebook
```

---

# Future Scope

- Deep Learning implementation
- Transformer-based fake review detection
- Real-time API integration
- Web application deployment
- Browser Extension
- Multi-language support
- Explainable AI dashboards

---

# Team

- Mayank Banerjee
- Souvagya Dey
- Suprovo Bose
- Milapan De


# Project Mentor

Mr. Ritesh Prasad
---

<p align="center">
Built using Python, Machine Learning, Natural Language Processing and XGBoost.
</p>
