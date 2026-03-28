# TechTrend 2026 Forecaster

An end-to-end data pipeline and interactive Streamlit dashboard for analyzing and predicting technology skill trends based on historical Reddit engagement data.

## Project Overview

This project processes large volumes of Reddit posts (from 2018 to 2025) to evaluate the trajectory of various technical skills. It calculates a composite **Weighted Score** for each skill, factoring in community engagement, sentiment, and trend momentum. Using state-of-the-art time-series forecasting models, the project predicts which skills will be in highest demand by 2026.

## End-to-End Pipeline

The analysis flows from text NLP processing to Machine Learning predictions:
1. **Preprocessing & Cleaning**: Ingests raw Reddit data, formats timestamps, and cleans up inconsistent keyword formats.
2. **Feature Engineering & NLP**: 
   - Utilizes `NLTK` to lemmatize text and strip stop words.
   - Leverages `vaderSentiment` to compute the emotional polarity (Sentiment Score) of the posts.
   - Computes a `Buzz` metric (Upvotes × Comments) and tracking Month-over-Month `Trend Growth`.
3. **Scoring**: Computes a final benchmark metric using a weighted scale (`Mention Volume (30%) + Buzz (40%) + Sentiment (20%) + Growth (10%)`), normalized using `MinMaxScaler`.
4. **Machine Learning Forecasting**:
   - Trains `Linear Regression`, `Prophet`, and `XGBoost` over the historical time-series data using temporal features (lags, sine/cosine cyclical encoding).
   - Validates model architecture on actual 2025 outputs to select the model with the lowest Mean Absolute Error (MAE) per skill.
   - Issues 2026 outlook predictions using the absolute best-fit model for each individual technical keyword.

## Repository Structure

```text
├── app.py                         # Streamlit Interactive Web Application
├── project_notebookFD.ipynb       # Main Data Science and ML Pipeline
├── plots/                         # Directory containing correlation and trend charts
├── model_evaluation.csv           # Model validation metrics & performances
├── monthly_features.csv           # Master historical dataset (2018-2025 features)
├── monthly_2026_predictions.csv   # Trajectory forecasting data for 2026
├── skill_forecast_2026.csv        # Summary scores outlook table
└── README.md                      # Documentation
```

## Setup & Local Usage

### 1. Dependencies

Run the following command to make sure you have the required science, modeling, and visualization libraries installed:

```bash
pip install pandas numpy matplotlib seaborn plotly nltk vaderSentiment scipy scikit-learn prophet xgboost streamlit
```

*Note: The NLTK packages (`punkt`, `stopwords`, `wordnet`) are automatically downloaded at runtime from the notebook context.*

### 2. Exploring the Analysis

Opening and running `project_notebookFD.ipynb` step-by-step will allow you to see the live breakdown of data extraction, graph generations, and modeling logs. This is only necessary if you add new raw metrics to the `Initial datasets` folder.

### 3. Launching the App Dashboard

The Streamlit app acts as an interactive dashboard visualizing everything in real-time. Start it by typing:

```bash
streamlit run app.py
```
This will open up a local server (`http://localhost:8501`) displaying historical trends, predictions, top skills rankings, and the model evaluation snapshot.
