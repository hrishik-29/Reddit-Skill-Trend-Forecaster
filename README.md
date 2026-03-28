# TechTrend 2026 Forecaster: Technical Walkthrough & Documentation

An end-to-end data pipeline and interactive Streamlit dashboard for analyzing and predicting technology skill trends based on historical Reddit engagement data. 

This project aims to answer a simple yet critical question: **Which tech skills will be the most valuable in 2026?** We answer this by empirically observing the rise and fall of developer discussions, rather than relying on gut feeling. 

---

## 📖 Table of Contents
1. [Introduction to the Methodology](#introduction-to-the-methodology)
2. [Step 1: Data Ingestion & Sanitization](#step-1-data-ingestion--sanitization)
3. [Step 2: NLP & Post-Level Feature Engineering](#step-2-nlp--post-level-feature-engineering)
4. [Step 3: Monthly Aggregation & Trend Engineering](#step-3-monthly-aggregation--trend-engineering)
5. [Step 4: The Weighted Score (The Metric That Matters)](#step-4-the-weighted-score-the-metric-that-matters)
6. [Step 5: Time-Series Prepping & Smoothing](#step-5-time-series-prepping--smoothing)
7. [Step 6: The "Algorithm Race" (Model Training & Validation)](#step-6-the-algorithm-race-model-training--validation)
8. [Step 7: 2026 Forecasting Engine](#step-7-2026-forecasting-engine)
9. [The Streamlit Dashboard](#the-streamlit-dashboard)
10. [Repository Structure & Setup](#repository-structure--setup)

---

## Introduction to the Methodology

Counting mentions of a technology isn't enough to predict its trajectory. A technology might be mentioned 10,000 times because of a severe outagelike bug, generating negative noise, while another technology is mentioned 2,000 times with massive community celebration and excitement. 

To solve this, our data pipeline breaks down conversational nuances into raw mathematics: **Sentiment, Engagement (Buzz), Frequency, and Momentum.** We fuse these into a single "Weighted Score" tracking system. 

Here is exactly how we go from raw Reddit posts to our 2026 predictions.

---

## Step 1: Data Ingestion & Sanitization

Our primary dataset (`Initial datasets/cleaned_final_dataset.csv`) consists of millions of Reddit posts scraped across various programming subreddits spanning from 2018 to 2025. 

**The Immediate Cleaning Steps:**
1. **Timestamp Normalization**: We convert the raw `created_utc` column into formal Pandas Datetime objects, dropping any completely malformed rows. From this, we extract strict `year`, `month`, and structured `year_month` string classifiers (e.g., "2023-08").
2. **Keyword Consolidation**: Because users type "RAG System", "retrieval augmented generation", and "Retrieval Augmented Generation", we use a custom mapping dictionary to dynamically squeeze all identical variants into a standardized `RAG system` keyword pool.
3. **Missing Value Processing**: Empty numeric `upvotes` or `comments_count` are imputed to `0`. Empty text bodies are safely parsed as empty strings.

---

## Step 2: NLP & Post-Level Feature Engineering

Before we can group data by month, we need to extract intelligence from *every single individual post*. We apply Natural Language Processing (NLP) to the post titles and bodies.

1. **Text Assembly & Cleaning**: We concatenate a post's Title and Body (`Title + " " + Body`) into one blob. We regex out all URLs (`http://...`) and non-alphabetical punctuation.
2. **Tokenization & Stop-Words Removal**: Using the `NLTK` library, we split the sentence into individual words ("tokens") and delete common filler words like "the", "and", "is". 
3. **Lemmatization**: We map variations of words back to their roots (e.g., "coding", "codes", "coded" all become "code") using `WordNetLemmatizer`.
4. **Sentiment Extraction (VADER)**: We pass the cleaned text payload into `vaderSentiment` (`SentimentIntensityAnalyzer()`). VADER specifically calculates an emotional polarity compound score for the post ranging from `-1.0` (extremely negative/angry) to `+1.0` (euphoric/happy). We save this as the `sentiment` feature.
5. **The Buzz Proxy Metric**: We create our first custom formula to measure raw conversational engagement scale. 
   >*Formula:* `buzz = upvotes * comments_count`
   > 
   >*Reasoning*: A post with 10 upvotes and 100 comments has a much deeper community discussion (Buzz: 1000) than a post with 100 upvotes but 0 comments (Buzz: 0). 

---

## Step 3: Monthly Aggregation & Trend Engineering

With every individual post carrying its own `sentiment` and `buzz`, we aggregate the data into monthly blocks. For every tech skill, for every single month, we group the rows and calculate:

1. **Mention Count**: The absolute number of posts about that skill in that specific month.
2. **Average Buzz**: The mean `buzz` score of all posts that month.
3. **Average Sentiment**: The mean `sentiment` compound score of all posts that month. 

**Calculating Momentum (Trend Growth):**
To see if a tech stack is accelerating or dying out, we look backward. We shift the data by one month (`prev_mentions = lag(1)`) and calculate the percentage growth.
> *Formula:* `trend_growth = (mention_count_current - mention_count_prev) / mention_count_prev`

*(Infinites or NaNs caused by dividing by zero are safely corrected to 0.0).*

---

## Step 4: The Weighted Score (The Metric That Matters)

Now we have our 4 core intermediate features: **Mentions, Buzz, Sentiment, and Growth**. 

Because these metrics operate on radically different numeric scales (mentions could be 5,000, while sentiment is capped at 1.0), we normalize them per keyword. We run `MinMaxScaler` exclusively grouped by the specific tech skill. This compresses the entire historical range of a skill's mentions into a `0.0` to `1.0` scale (`_norm`). 

Finally, we apply our custom weighted composite formula. 

**The Official Weighted Score Formula:**
> `Weighted Score = (`
> `    0.30 * mention_count_norm + `
> `    0.40 * avg_buzz_norm + `
> `    0.20 * avg_sentiment_norm + `
> `    0.10 * trend_growth_norm`
> `) * 100`

We multiply by 100 (and clip bounds between 0 and 100) strictly to make the final score easily digestible for human viewers out of a familiar 0-100 scale. 
> *Logic applied*: High conversational `Buzz` (40%) and sheer volume `Mentions` (30%) drive the bulk of the score, penalized heavily if `Sentiment` (20%) is negative, and slightly padded by acceleration `Growth` (10%).

---

## Step 5: Time-Series Prepping & Smoothing

Before ML models can learn from this historical data, we need to prepare the timelines to be strictly continuous. 
1. **Resampling**: We force the index to formal "Month Start" (`MS`) Pandas frequencies. We interpolate any missing intermediate months softly using neighboring data.
2. **Signal Noise Reduction (Smoothing)**: Time series data is famously erratic. We apply a 3-month rolling moving average to our score.
   > *Formula:* `smoothed_score = 3_month_rolling_mean(weighted_score)`
3. **Cyclical Temporal Feature Extraction**: ML natively struggles to understand that December (Month 12) is next to January (Month 1). So we geometrically map time onto a circle utilizing sine and cosine waves:
   > *Formula:* `month_sin = sin(2 * π * month / 12)`
   > *Formula:* `month_cos = cos(2 * π * month / 12)`
4. **Autoregressive Lags**: We feed the model memories of its recent past to help it predict the future. We create two new columns:
   > `lag_1 = smoothed_score 1 month ago`
   > `lag_3 = smoothed_score 3 months ago`

---

## Step 6: The "Algorithm Race" (Model Training & Validation)

We do not trust one single algorithm to predict everything. A technology that has been stable for 8 years acts completely differently than a technology that exploded virally 6 months ago. 

For **every single individual tech skill** (e.g., Python, RAG, Rust, NextJS), the system triggers a private algorithm race:
1. **The Time Split**: We train the models entirely using `2018–2024` data. We set `2025` data strictly aside to serve as a blind test environment.
2. **The Racers**:
    - **Linear Regression**: Excellent at projecting stable, highly linear traditional uptrends.
    - **Prophet (Meta)**: Heavy duty additive time-series framework handling weekly/yearly structural seasonality. 
    - **XGBoost Regressor**: Gradient-boosted decision trees that excel at mapping chaotic non-linear momentum structures.
    - **Persistence Baseline**: The "dumb" model that naively guesses the previous month's value will just repeat identically.
3. **The Validation & Coronation**: We force all 4 mathematical models to predict all 12 months for 2025 based on the test variables. We compare the predictions identically against what actually happened in 2025 by measuring the **Mean Absolute Error (MAE)**.
4. **Winner Selected**: Whichever model logged the absolute lowest error margin (e.g., XGBoost achieved a smaller MAE than Prophet for the keyword "Rust") is officially crowned the "Best Model" exclusively for that specific skill. This evaluation is stored in `model_evaluation.csv`.

---

## Step 7: 2026 Forecasting Engine

Now that we know dynamically which algorithm is mathematically best suited to map out each specific skill's unique flavor of momentum, we deploy the final forecasting stage.

1. **Refitting**: We take the "Best Model" for a skill and retrain it utilizing the **entire dataset** up to the exact end of 2025. 
2. **Iterative Rolling Forecasts**: We instruct the model to simulate 12 sequential months natively spanning 2026. 
   - Because we rely heavily on Autoregressive Lags (`lag_1`, `lag_3`), projecting Month 2 inherently requires the output of Month 1. We accomplish this using an iterative loop. Month 1 is predicted using December 2025 real lags. Month 1's prediction is then pushed physically into the memory frame to act as the `lag_1` variable for predicting Month 2, carrying forward mechanically until December 2026 is solved.
3. **File Output Generation**: The trajectory predictions are formally exported into `monthly_2026_predictions.csv` while the average 2026 scores are dumped to `skill_forecast_2026.csv`. 

---

## The Streamlit Dashboard

`app.py` is the capstone presentation layer. It natively ingests our four exported `.csv` intermediate files generated by the backend pipeline previously. 

**Features:**
- **Sidebar Skill Selector**: Loads the dynamic list of analyzed technologies. 
- **Graphing (`Plotly`)**: Immediately extracts the `hist_df` (Historical 2018-2025 timeline) and graphs it seamlessly connecting into `forecast_df` (Dashed 2026 line).
- **Metric Cards**: Exposes the average 2026 aggregate score, the specifically selected Best Model, and the historical error (MAE / R²) to objectively validate the projection's reliability.
- **Categorical Logic (Recommendations)**: The application checks the 2026 score value. 
  - `Score >= 80` ➔ **High Priority**
  - `Score >= 60` ➔ **Consider Investing**
  - `Score >= 40` ➔ **Monitor Emerging Demand**
  - `Score < 40` ➔ **Watchlist**

---

## Repository Structure & Setup

```text
├── app.py                         # The Streamlit Interactive Web Application
├── project_notebookFD.ipynb       # The complete Data Science and ML Pipeline
├── plots/                         # Generated visualizations (trends, heatmaps, distributions)
├── model_evaluation.csv           # Model validation metrics (MAE, RMSE, R2) for every skill
├── monthly_features.csv           # Master historical dataset (2018-2025 features & scores)
├── monthly_2026_predictions.csv   # The month-by-month trajectory forecasting for 2026
├── skill_forecast_2026.csv        # Final summary of 2026 predictions and model tracking
└── README.md                      # Project Documentation
```

### 1. Dependencies Setup
Ensure you have all the required scientific libraries installed.
```bash
pip install pandas numpy matplotlib seaborn plotly nltk vaderSentiment scipy scikit-learn prophet xgboost streamlit
```

### 2. Exploring the Full Pipeline
You can open `project_notebookFD.ipynb` in your preferred Jupyter environment to individually run every mathematical processing block, print diagnostic graphs, and generate fresh local metric `.csv` artifacts. Note that `NLTK` vocabularies compile dynamically on the first run.

### 3. Launching the App Dashboard
To explore the calculated time-series data locally via the user interface:
```bash
streamlit run app.py
```
This automatically initiates a local web server (usually `localhost:8501`) granting you interactive insight generation logic matching real historical context.
