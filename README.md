# NexForge — Predictive Maintenance System

An AI-powered predictive maintenance system that uses an optimized XGBoost model to estimate the probability of industrial machine failure from operating sensor data.

**Live Demo:** https://predictive-maintenance-ann.vercel.app

## Overview

NexForge demonstrates how machine learning can support predictive maintenance in industrial environments. Instead of relying only on scheduled maintenance or reacting after a machine fails, the system analyzes operating parameters and predicts the likelihood of failure.

The application provides a dashboard where users can enter machine sensor readings and receive a failure prediction with its estimated probability.

## Key Features

- **Machine Failure Prediction:** Uses an optimized XGBoost classification model.
- **Failure Probability:** Displays the estimated probability of machine failure.
- **Machine Type Selection:** Supports L, M, and H machine types.
- **Sensor-Based Assessment:** Accepts air temperature, process temperature, rotational speed, torque, and tool wear.
- **User Authentication:** Firebase Authentication with email/password and Google sign-in.
- **Interactive Dashboard:** Provides a dashboard for machine assessment and prediction results.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React, Vite |
| Styling | CSS |
| Backend | Python, FastAPI |
| Machine Learning | Optimized XGBoost |
| Data Processing | Pandas |
| Model Serialization | Joblib |
| Authentication | Firebase Authentication |
| Frontend Deployment | Vercel |
| Backend Deployment | Render |
| Dataset | AI4I 2020 Predictive Maintenance Dataset |

## Machine Learning Model

The project uses an optimized XGBoost model trained for binary classification.

**Input features:**
- Air temperature (K)
- Process temperature (K)
- Rotational speed (rpm)
- Torque (Nm)
- Tool wear (min)
- Machine type (L, M, H)

**Output:**
- Failure probability
- Failure probability percentage
- Prediction: `MACHINE FAILURE` or `NO MACHINE FAILURE`

The trained model is stored as `saved_models/optimized_xgboost.pkl` and loaded by the backend.

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**, which contains industrial machine operating data and machine-failure labels.

Dataset source: [AI4I 2020 Predictive Maintenance Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

## System Architecture

1. The user accesses the NexForge frontend.
2. Firebase Authentication manages user sign-in.
3. The user enters machine operating parameters.
4. The React frontend sends the input data to the FastAPI backend.
5. The backend prepares the input features and passes them to the trained XGBoost model.
6. The model returns a failure probability and prediction.
7. The frontend displays the result on the dashboard.

## Deployment

- **Frontend:** Deployed on Vercel.
- **Backend:** Deployed on Render.
- **Authentication:** Firebase Authentication.

The hosted backend may take some time to respond to its first request after inactivity because the project uses Render's free instance.

## Project Status

The frontend and backend are deployed, Firebase authentication is integrated, and the machine-failure prediction workflow has been tested successfully.

---

**NexForge — Turning Industrial Data into Predictive Insights.**