# 🪐 Kepler Planetary Signal Review System (PSRS)

AI-assisted scientific review prioritization and historical signal retrieval system inspired by NASA's Kepler exoplanet discovery mission.

---

# Overview

The Kepler Planetary Signal Review System (PSRS) is a full-stack AI application designed to assist scientific review of planetary signals detected by the Kepler Space Telescope.

The system combines machine learning classification, confidence-based prioritization, semantic similarity search, authentication, database-backed review management, and a web interface to reduce manual review workload while maintaining high recall for potentially confirmed exoplanets.

Key capabilities include:

* Multi-class exoplanet signal classification
* Confidence-based review prioritization
* Semantic retrieval of historically similar signals
* JWT-based user authentication
* Database-backed prediction tracking
* Full CRUD review workflow
* REST API deployment with FastAPI
* Interactive frontend built with HTML, CSS, and JavaScript
* Gradio deployment for ML demonstration

---

# Problem Statement

Kepler mission signals are typically classified into:

* CONFIRMED
* CANDIDATE
* FALSE POSITIVE

Scientific validation requires manual review of thousands of detected signals, creating significant operational overhead.

Challenges include:

* Large review volume
* Limited scientific resources
* Need to preserve high recall for potentially confirmed exoplanets
* Efficient prioritization of scientific review effort

---

# Solution Architecture

## 1. Multi-Class Classification

Three machine learning models were trained and evaluated:

* Logistic Regression
* Random Forest
* XGBoost

Based on classification performance and recall for confirmed planets, XGBoost was selected as the final production model.

---

## 2. Deployment-Ready ML Pipeline

A Scikit-learn pipeline was built using:

* SimpleImputer
* XGBoost
* GridSearchCV

This ensures consistent preprocessing and inference across:

* Training notebooks
* Gradio interface
* FastAPI services

---

## 3. Confidence-Based Review Prioritization

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

## 4. Semantic Historical Signal Retrieval

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

## 5. Authentication & User Management

The application implements JWT-based authentication using FastAPI security utilities.

Features include:

* User Registration
* User Login
* JWT Token Authentication
* Protected API Endpoints
* User-specific Prediction History
* Logout Functionality

Each prediction is associated with the authenticated user, ensuring prediction records remain isolated between users.

---

## 6. Human-in-the-Loop Review Workflow

Predictions are persisted in a SQLite database using SQLAlchemy.

The workflow enables:

* Storage of prediction history
* Retrieval of previous predictions
* Review status updates
* Deletion of obsolete records

This simulates a scientific review pipeline where machine learning assists but does not replace human validation.

---

# Full Stack Application Workflow

1. User registers an account.
2. User logs into the platform.
3. JWT access token is generated and stored on the client.
4. User submits planetary signal parameters.
5. XGBoost model generates a classification prediction.
6. Confidence-based prioritization assigns a review recommendation.
7. Prediction is stored in a SQLite database.
8. User can view, update, or delete previous predictions.
9. Historical similarity search retrieves semantically related planetary signals using FAISS.

---

# Results

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

# Frontend

A lightweight frontend was developed using:

* HTML5
* CSS3
* JavaScript (ES6)

Pages include:

* Register
* Login
* Dashboard
* New Prediction
* My Predictions
* Historical Signals

The frontend communicates with FastAPI REST endpoints using the Fetch API and JWT-based authorization headers.

---

# Database Design

Prediction records are stored using SQLAlchemy ORM.

Stored attributes include:

* Prediction ID
* Prediction Class
* Confidence Score
* Priority Level
* Review Status
* User Association

Supported operations:

* Create Prediction
* Read Predictions
* Update Review Status
* Delete Prediction

This enables a complete CRUD workflow for scientific review management.

---

# API Endpoints

## Authentication

### POST `/register`

Creates a new user account.

### POST `/login`

Authenticates a user and returns a JWT access token.

### GET `/me`

Returns information about the authenticated user.

---

## Predictions

### POST `/predict`

Predicts signal disposition and stores prediction records.

Returns:

* Predicted Class
* Confidence Score
* Priority Level
* Review Recommendation

### GET `/predictions`

Retrieves prediction history for the authenticated user.

### PATCH `/predictions/{prediction_id}`

Updates review status for an existing prediction record.

### DELETE `/predictions/{prediction_id}`

Deletes a prediction record.

---

## Historical Retrieval

### POST `/history`

Retrieves semantically similar historical planetary signals.

Returns:

* Similar Planets
* Historical Dispositions
* Signal Summaries

---

# Tech Stack

## Machine Learning

* Scikit-learn
* XGBoost
* Pandas
* NumPy

## Semantic Search

* Sentence Transformers
* FAISS

## Backend

* FastAPI
* SQLAlchemy
* SQLite
* JWT Authentication
* Pydantic

## Frontend

* HTML5
* CSS3
* JavaScript

## Deployment

* Gradio
* Hugging Face Spaces

---

# Project Structure

```text
Kepler-SRS/

├── frontend/
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── new_predictions.html
│   ├── my_predictions.html
│   ├── historical_signals.html
│   ├── styles.css
│   └── js/

├── routers/
│   ├── auth_router.py
│   ├── prediction_router.py
│   └── history_router.py

├── services/
│   ├── predictor.py
│   └── retrieval.py

├── model_files/
│   ├── finalModel.joblib
│   ├── faiss_index.bin
│   ├── label_encoder.joblib
│   ├── planet_texts.joblib
│   └── history_df.csv

├── notebooks/
│   └── Kepler_SRS.ipynb

├── auth.py
├── database.py
├── models.py
├── schemas.py
├── main.py
└── app.py
```

---

# Deployment

### Machine Learning Demo

Hugging Face Spaces:

https://huggingface.co/spaces/Harman1010/Kepler-PSRS

### Full Stack Version

FastAPI + HTML/CSS/JavaScript application with authentication, CRUD operations, database persistence, and historical signal retrieval.

---

# Future Improvements

* PostgreSQL integration
* Role-based authorization
* Docker deployment
* Scientist feedback collection
* Active learning for model retraining
* Cloud-native deployment on Render
* Analytics dashboard for review monitoring
* Automated review recommendation tracking

---
