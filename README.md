🪐 Welcome to Astronomy with Machine Learning

Kepler Exoplanet Signal Review System

Overview

In this project, we built a machine learning–based signal review system to support NASA’s Kepler exoplanet discovery mission.
The objective is to reduce the manual workload of scientists while maintaining a high recall of confirmed exoplanets.

Problem

During the Kepler mission, detected signals are initially classified as:

CONFIRMED

CANDIDATE

FALSE POSITIVE

After this initial classification, scientists manually analyze signals to verify true exoplanets.
This manual review process is time-consuming and costly, especially given the large number of detected signals.

Solution

1️. Multi-Class Classification

We trained and compared three machine learning models:

Logistic Regression (Baseline)
Random Forest (Ensemble)
XGBoost (Fast, industry-relevant ensemble model)

Based on overall performance and recall for confirmed planets, XGBoost was selected as the final model.

2️. Threshold-Based System Design

Instead of relying only on predicted class labels, we designed a threshold-based decision system using the model’s predicted probability for the CONFIRMED class.

This converts the classifier into a decision-support system for prioritizing human review.

Results

Threshold selected: 0.3

Recall (CONFIRMED planets): ~89%

Signals sent for review: ~30%

Manual workload reduction: ~70%

This demonstrates a strong balance between scientific safety and operational efficiency.

Deployment

The final system is deployed as an interactive Gradio application that:

Accepts signal features
Outputs class probabilities
Recommends Send for Review / Skip decisions

🔗 Live Demo: https://huggingface.co/spaces/Harman1010/Kepler-PSRS

