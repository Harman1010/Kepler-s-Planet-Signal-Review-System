# 🪐 Kepler Planetary Signal Review System (PSRS)

AI-assisted scientific review prioritization and historical signal retrieval system inspired by NASA's Kepler exoplanet discovery mission.

---

## Overview

The Kepler Planetary Signal Review System (PSRS) is an end-to-end machine learning application designed to assist scientific review of planetary signals detected by the Kepler Space Telescope.

The system combines machine learning classification, confidence-based prioritization, semantic similarity search, and database-backed review management to reduce manual workload while maintaining high recall for potential exoplanets.

Key capabilities include:

* Multi-class exoplanet signal classification
* Confidence-based review prioritization
* Semantic retrieval of historically similar signals
* Human-in-the-loop review workflow
* Database-backed prediction tracking
* REST API deployment with FastAPI
* Interactive deployment with Gradio

---

## Problem Statement

Kepler mission signals are typically classified into:

* CONFIRMED
* CANDIDATE
* FALSE POSITIVE

Scientific validation requires manual review of thousands of detected signals, creating significant operational overhead.

Challenges include:

* Large review volume
* Limited scientific resources
* Need to preserve high recall for potentially confirmed exoplanets

---

## Solution Architecture

### 1. Multi-Class Classification

Three machine learning models were trained and evaluated:

* Logistic Regression
* Random Forest
* XGBoost

Based on classification performance and recall for confirmed planets, XGBoost was selected as the final production model.

---

### 2. Deployment-Ready ML Pipeline

A Scikit-learn pipeline was built using:

* SimpleImputer
* XGBoost
* GridSearchCV

This ensures consistent preprocessing and inference across:

* Training notebooks
* Gradio interface
* FastAPI services

---

### 3. Confidence-Based Review Prioritization

Instead of relying solely on class predictions, the system uses the predicted probability of the CONFIRMED class to prioritize scientific review.

Priority levels:

* Critical
* High
* Medium
* Low

Review recommendations:

* Immediate Review
* Scientist Validation
* Review Queue
* Auto-Filtered

This transforms the classifier into a decision-support system rather than a simple prediction engine.

---

### 4. Semantic Historical Signal Retrieval

To provide contextual scientific insight, the system retrieves historically similar planetary signals using:

* Sentence Transformers
* FAISS Vector Search

Similarity retrieval is based on:

* Orbital Period
* Transit Depth
* Planet Radius
* Stellar Temperature

For each incoming signal, the system returns:

* Similar historical planets
* Historical classifications
* Signal summaries

---

### 5. Human-in-the-Loop Review Workflow

Predictions are persisted in a SQLite database using SQLAlchemy.

The workflow enables:

* Storage of prediction history
* Retrieval of previous predictions
* Review status updates
* Deletion of obsolete records

This simulates a scientific review pipeline where machine learning assists but does not replace human validation.

---

## Results

| Metric                       | Value |
| ---------------------------- | ----- |
| Selected Threshold           | 0.30  |
| Recall (CONFIRMED)           | ~89%  |
| Signals Sent for Review      | ~30%  |
| Estimated Workload Reduction | ~70%  |

The system achieves a strong balance between:

* Scientific safety
* Operational efficiency
* Review scalability

---

## API Endpoints

### POST `/predict`

Predicts signal disposition and stores prediction records.

Returns:

* Predicted Class
* Confidence Score
* Priority Level
* Review Recommendation

---

### POST `/history`

Retrieves semantically similar historical planetary signals.

Returns:

* Similar Planets
* Historical Dispositions
* Signal Summaries

---

### GET `/predictions`

Retrieves stored prediction history.

---

### PATCH `/predictions/{prediction_id}`

Updates review status for an existing prediction record.

---

### DELETE `/predictions/{prediction_id}`

Deletes a prediction record from the database.

---

## Tech Stack

### Machine Learning

* Scikit-learn
* XGBoost
* Pandas
* NumPy

### Semantic Search

* Sentence Transformers
* FAISS

### Backend

* FastAPI
* SQLAlchemy
* SQLite

### Deployment

* Gradio
* Hugging Face Spaces

---

## Deployment

Live Demo:

https://huggingface.co/spaces/Harman1010/Kepler-PSRS

---

## Future Improvements

* PostgreSQL integration
* Reviewer authentication and authorization
* Scientist feedback collection
* Active learning for model retraining
* Dashboard-based review management
* Cloud deployment with Docker
