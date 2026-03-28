import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

ST_DATA_DIR = Path('.')

@st.cache_data(show_spinner=False)
def load_monthly_features() -> pd.DataFrame:
    path = ST_DATA_DIR / 'monthly_features.csv'
    if path.exists():
        df = pd.read_csv(path, parse_dates=['month_start'])
    else:
        df = pd.DataFrame()
    return df

@st.cache_data(show_spinner=False)
def load_monthly_forecast() -> pd.DataFrame:
    path = ST_DATA_DIR / 'monthly_2026_predictions.csv'
    if path.exists():
        df = pd.read_csv(path, parse_dates=['date'])
    else:
        df = pd.DataFrame()
    return df

@st.cache_data(show_spinner=False)
def load_skill_scores() -> pd.DataFrame:
    path = ST_DATA_DIR / 'skill_forecast_2026.csv'
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame()
    return df

@st.cache_data(show_spinner=False)
def load_model_evaluation() -> pd.DataFrame:
    path = ST_DATA_DIR / 'model_evaluation.csv'
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame()
    return df

st.set_page_config(page_title='TechTrend 2026 Forecaster', layout='wide')
st.title('TechTrend 2026 Forecaster')

with st.sidebar:
    st.header('Explorer Controls')

monthly_features = load_monthly_features()
monthly_forecast = load_monthly_forecast()
skill_scores = load_skill_scores()
model_eval = load_model_evaluation()

if monthly_features.empty or skill_scores.empty:
    st.error('Forecast artefacts not found. Please generate the notebook outputs before launching the app.')
    st.stop()

monthly_features['month_start'] = pd.to_datetime(monthly_features['month_start'], errors='coerce')
monthly_features.dropna(subset=['month_start'], inplace=True)

skills = sorted(skill_scores['keyword'].unique())
selected_skill = st.sidebar.selectbox('Select a skill', skills)

if model_eval.empty:
    st.sidebar.warning('Model evaluation metrics unavailable. Showing forecasts only.')
else:
    best_lookup = (
        model_eval.sort_values('MAE')
        .groupby('keyword', as_index=False)
        .first()
        .rename(columns={'model': 'best_model', 'MAE': 'best_MAE', 'RMSE': 'best_RMSE', 'R2': 'best_R2'})
    )
    skill_scores = skill_scores.merge(best_lookup, on='keyword', how='left')

hist_df = monthly_features[monthly_features['keyword'] == selected_skill].copy()
forecast_df = monthly_forecast[monthly_forecast['keyword'] == selected_skill].copy()
forecast_df.rename(columns={'date': 'month_start'}, inplace=True)

fig = go.Figure()
if not hist_df.empty:
    fig.add_trace(
        go.Scatter(
            x=hist_df['month_start'],
            y=hist_df['weighted_score'],
            mode='lines',
            name='Historical 2018-2025',
            line=dict(color='#1f77b4', width=3)
        )
    )

if not forecast_df.empty:
    fig.add_trace(
        go.Scatter(
            x=forecast_df['month_start'],
            y=forecast_df['predicted_score'],
            mode='lines+markers',
            name='Forecast 2026',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=8)
        )
    )

fig.update_layout(
    margin=dict(l=20, r=20, t=40, b=40),
    xaxis_title='Month',
    yaxis_title='Weighted Score',
    legend_title='Series'
)

st.plotly_chart(fig, use_container_width=True)

score_row = skill_scores[skill_scores['keyword'] == selected_skill]
if not score_row.empty:
    score_value = float(score_row['score_2026'].iloc[0])
    best_model_name = score_row.get('best_model', pd.Series(['NA'])).iloc[0]
    best_mae = score_row.get('best_MAE', pd.Series([float('nan')])).iloc[0]
    best_rmse = score_row.get('best_RMSE', pd.Series([float('nan')])).iloc[0]
    best_r2 = score_row.get('best_R2', pd.Series([float('nan')])).iloc[0]
else:
    score_value = float('nan')
    best_model_name = 'NA'
    best_mae = float('nan')
    best_rmse = float('nan')
    best_r2 = float('nan')

metric_cols = st.columns(4)
metric_cols[0].metric('2026 Score', f"{score_value:.1f}/100" if pd.notna(score_value) else 'NA')
metric_cols[1].metric('Best Model', best_model_name if best_model_name else 'NA')
metric_cols[2].metric('Best MAE', f"{best_mae:.2f}" if pd.notna(best_mae) else 'NA')
metric_cols[3].metric('Best R²', f"{best_r2:.2f}" if pd.notna(best_r2) else 'NA')

if pd.notna(score_value):
    if score_value >= 80:
        recommendation = 'High Priority'
    elif score_value >= 60:
        recommendation = 'Consider Investing'
    elif score_value >= 40:
        recommendation = 'Monitor Emerging Demand'
    else:
        recommendation = 'Watchlist'
else:
    recommendation = 'Insufficient data to rate.'

st.write(f"**Recommendation:** {recommendation}")

if not forecast_df.empty:
    st.subheader('2026 Monthly Forecast')
    st.dataframe(
        forecast_df[['month_start', 'predicted_score']]
        .rename(columns={'month_start': 'Month', 'predicted_score': 'Predicted Score'})
        .round({'Predicted Score': 2}),
        use_container_width=True,
    )

if not hist_df.empty:
    st.subheader('Historical Weighted Scores (2018-2025)')
    st.dataframe(
        hist_df[['month_start', 'weighted_score']]
        .rename(columns={'month_start': 'Month', 'weighted_score': 'Weighted Score'})
        .round({'Weighted Score': 2}),
        use_container_width=True,
    )

if not model_eval.empty:
    st.subheader('Model Evaluation Snapshot')
    eval_skill = (
        model_eval[model_eval['keyword'] == selected_skill]
        .sort_values('MAE')
        .reset_index(drop=True)
    )
    st.dataframe(
        eval_skill[['model', 'MAE', 'RMSE', 'R2']].round({'MAE': 2, 'RMSE': 2, 'R2': 3}),
        use_container_width=True,
    )

if not skill_scores.empty:
    st.subheader('2026 Skill Outlook')
    top_n = st.slider('Show top N skills by forecast score', min_value=5, max_value=min(25, len(skill_scores)), value=min(10, len(skill_scores)))
    top_skills = skill_scores.sort_values('score_2026', ascending=False).head(top_n)
    fig_bar = px.bar(
        top_skills,
        x='score_2026',
        y='keyword',
        color='best_model' if 'best_model' in top_skills.columns else None,
        orientation='h',
        labels={'score_2026': 'Average 2026 Score', 'keyword': 'Skill', 'best_model': 'Best Model'},
    )
    fig_bar.update_layout(margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.dataframe(
        top_skills[['keyword', 'score_2026', 'best_model', 'best_R2'] if 'best_model' in top_skills.columns else ['keyword', 'score_2026']]
        .rename(columns={'keyword': 'Skill', 'score_2026': 'Avg 2026 Score', 'best_model': 'Best Model', 'best_R2': 'Best R²'})
        .round({'Avg 2026 Score': 2, 'Best R²': 3})
        ,
        use_container_width=True,
    )
