# 🪐 Kepler Planetary Signal Review System (PSRS)

AI-assisted scientific review prioritization system for Kepler exoplanet signal analysis.

---

## Overview

In this project, we built a machine learning–based signal review system inspired by NASA’s Kepler exoplanet discovery mission.

The goal of the system is to reduce manual scientific review workload while maintaining high recall for potential exoplanets.

The project combines:

- Multi-class machine learning classification
- Confidence-based review prioritization
- Semantic historical signal retrieval
- FastAPI and Gradio deployment

---

## Problem

During the Kepler mission, detected signals are initially classified as:

- CONFIRMED
- CANDIDATE
- FALSE POSITIVE

Scientists must manually review thousands of detected signals to identify true exoplanets.

This process is:
- Time-consuming
- Resource-intensive
- Difficult to scale efficiently

---

## Solution

### 1️. Multi-Class Classification

We trained and compared three machine learning models:

- Logistic Regression
- Random Forest
- XGBoost

Based on overall performance and recall for confirmed planets, XGBoost was selected as the final deployment model.

---

### 2️. Deployment-Ready ML Pipeline

A Scikit-learn pipeline was created using:

- SimpleImputer
- XGBoost
- GridSearchCV optimization

This ensured consistent preprocessing and inference across:
- Notebook experimentation
- Gradio deployment
- FastAPI deployment

---

### 3️. Confidence-Based Review Prioritization

Instead of relying only on predicted class labels, the system uses predicted probability for the `CONFIRMED` class to drive review decisions.

Signals are categorized into:
- Critical
- High
- Medium
- Low priority

This converts the classifier into a scientific decision-support system.

---

### 4️. Semantic Historical Signal Retrieval

The project also includes a semantic similarity retrieval system using:

- Sentence Transformers
- FAISS vector search

For every input signal, the system retrieves historically similar planetary signals based on:

- Orbital Period
- Transit Depth
- Planet Radius
- Stellar Temperature

This provides contextual historical comparison for scientific interpretation.

---

## Results

- Threshold selected: `0.3`
- Recall for CONFIRMED planets: `~89%`
- Signals forwarded for review: `~30%`
- Estimated manual workload reduction: `~70%`

The system demonstrates a strong balance between:
- Scientific safety
- Operational efficiency

---

## API Endpoints

### `/predict`

Returns:
- Predicted class
- Confidence score
- Review priority
- Review recommendation

### `/history`

Returns:
- Similar historical planetary signals
- Historical dispositions
- Planet summaries

---

## Tech Stack

- Python
- Scikit-learn
- XGBoost
- FastAPI
- Gradio
- Sentence Transformers
- FAISS
- Pandas
- NumPy

---

## Deployment

The final system was deployed using:
- Gradio
- Hugging Face Spaces
- FastAPI backend architecture

🔗 Live Demo:  
https://huggingface.co/spaces/Harman1010/Kepler-PSRS

---
