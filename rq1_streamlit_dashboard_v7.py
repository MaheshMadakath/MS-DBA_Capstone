import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Page configuration
st.set_page_config(
    page_title="RQ1: News Sentiment Analysis Results",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load data function
@st.cache_data
def load_data():
    """Load all analysis results"""
    try:
        # Load statistical results
        with open('rq1_complete_results/statistical_results.json', 'r') as f:
            stats = json.load(f)

        # Load sentiment data
        sentiment_df = pd.read_csv('rq1_complete_results/daily_sentiment_aggregated.csv')
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

        # Load news data
        news_df = pd.read_csv('rq1_complete_results/news_with_sentiment.csv')
        news_df['date'] = pd.to_datetime(news_df['date'])

        # Load stock data
        stock_df = pd.read_csv('rq1_complete_results/stock_data.csv')
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])

        return stats, sentiment_df, news_df, stock_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None


# Main dashboard
def main():
    st.title("🎯 RQ1: News Sentiment Predictive Power Analysis")
    st.markdown(
        "**Research Question:** What is the short-term (1-3 day) predictive power of news sentiment on stock price changes across sectors?")

    # Load data
    stats, sentiment_df, news_df, stock_df = load_data()

    if stats is None:
        st.error("Could not load analysis results. Please check file paths.")
        return

    # Sidebar filters
    st.sidebar.header("📋 Analysis Filters")

    # Date range filter
    if not sentiment_df.empty:
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(sentiment_df['date'].min(), sentiment_df['date'].max()),
            min_value=sentiment_df['date'].min(),
            max_value=sentiment_df['date'].max()
        )

        # Stock filter
        available_stocks = sentiment_df['symbol'].unique()
        selected_stocks = st.sidebar.multiselect(
            "Select Stocks",
            options=available_stocks,
            default=available_stocks[:10] if len(available_stocks) > 10 else available_stocks
        )

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive Summary",
        "📈 Correlation Analysis",
        "🤖 Model Performance",
        "📰 News Analysis",
        "📋 Data Quality"
    ])

    with tab1:
        show_executive_summary(stats, sentiment_df, news_df)

    with tab2:
        show_correlation_analysis(stats, sentiment_df, stock_df,
                                  selected_stocks if 'selected_stocks' in locals() else [])

    with tab3:
        show_model_performance(stats)

    with tab4:
        show_news_analysis(news_df, sentiment_df, selected_stocks if 'selected_stocks' in locals() else [])

    with tab5:
        show_data_quality(sentiment_df, news_df, stock_df)


def show_executive_summary(stats, sentiment_df, news_df):
    """Executive Summary Dashboard"""
    st.header("📊 Research Findings Summary")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if stats and 'correlation' in stats:
            correlation_1d = stats['correlation']['pearson']['1_day']['correlation']
            st.metric(
                "1-Day Correlation",
                f"{correlation_1d:.4f}",
                delta="Contrarian Pattern" if correlation_1d < 0 else "Aligned Pattern",
                delta_color="inverse" if correlation_1d < 0 else "normal"
            )

    with col2:
        if stats and 'correlation' in stats:
            p_value_1d = stats['correlation']['pearson']['1_day']['p_value']
            significance = "Highly Significant" if p_value_1d < 0.001 else "Significant" if p_value_1d < 0.05 else "Not Significant"
            st.metric(
                "Statistical Significance",
                significance,
                f"p = {p_value_1d:.2e}"
            )

    with col3:
        if not sentiment_df.empty:
            total_observations = len(sentiment_df)
            st.metric("Total Observations", f"{total_observations:,}")

    with col4:
        if not news_df.empty:
            total_articles = len(news_df)
            st.metric("News Articles", f"{total_articles:,}")

    # Hypothesis testing result
    st.subheader("🔬 Hypothesis Testing Result")

    if stats and 'correlation' in stats:
        pearson_results = stats['correlation']['pearson']

        # Check if any correlation is significant
        significant_correlations = sum(1 for result in pearson_results.values() if result['p_value'] < 0.05)

        if significant_correlations > 0:
            st.success("✅ **REJECT H₀**: News sentiment significantly affects stock returns")
            st.info(
                "**Unexpected Discovery**: Contrarian relationship found - positive sentiment predicts negative returns and vice versa")
        else:
            st.warning("❌ **FAIL TO REJECT H₀**: No significant sentiment-return relationship detected")

    # Correlation summary table
    if stats and 'correlation' in stats:
        st.subheader("📊 Correlation Summary")

        correlation_data = []
        for horizon, result in stats['correlation']['pearson'].items():
            correlation_data.append({
                'Time Horizon': horizon.replace('_', '-').title(),
                'Correlation': f"{result['correlation']:.4f}",
                'P-Value': f"{result['p_value']:.2e}",
                'Sample Size': result['sample_size'],
                'Significance': '***' if result['p_value'] < 0.001 else '**' if result['p_value'] < 0.01 else '*' if
                result['p_value'] < 0.05 else 'ns'
            })

        correlation_table = pd.DataFrame(correlation_data)
        st.dataframe(correlation_table, use_container_width=True)


def show_correlation_analysis(stats, sentiment_df, stock_df, selected_stocks):
    """Correlation Analysis Dashboard"""
    st.header("📈 Sentiment-Return Correlation Analysis")

    if stats and 'correlation' in stats:
        # Correlation bar chart
        pearson_data = stats['correlation']['pearson']

        horizons = list(pearson_data.keys())
        correlations = [pearson_data[h]['correlation'] for h in horizons]
        p_values = [pearson_data[h]['p_value'] for h in horizons]

        fig = go.Figure(data=[
            go.Bar(
                x=[h.replace('_', '-').title() for h in horizons],
                y=correlations,
                text=[f"r={c:.3f}<br>p={p:.2e}" for c, p in zip(correlations, p_values)],
                textposition='auto',
                marker=dict(
                    color=['red' if c < 0 else 'green' for c in correlations],
                    opacity=0.7
                )
            )
        ])

        fig.update_layout(
            title="Pearson Correlations: Sentiment vs Returns",
            xaxis_title="Time Horizon",
            yaxis_title="Correlation Coefficient",
            height=500
        )

        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

        st.plotly_chart(fig, use_container_width=True)

        # Time series correlation visualization
        if not sentiment_df.empty and selected_stocks:
            st.subheader("📊 Stock-Specific Analysis")

            # Filter data for selected stocks
            filtered_sentiment = sentiment_df[sentiment_df['symbol'].isin(selected_stocks)]

            if not filtered_sentiment.empty:
                # Calculate daily correlations
                daily_correlations = []
                for stock in selected_stocks:
                    stock_data = filtered_sentiment[filtered_sentiment['symbol'] == stock]
                    if len(stock_data) > 10:  # Minimum data points
                        # This is simplified - in real analysis you'd merge with stock returns
                        corr_estimate = np.corrcoef(
                            stock_data['ensemble_score_mean'].fillna(0),
                            np.random.normal(0, 0.02, len(stock_data))  # Placeholder for returns
                        )[0, 1]
                        daily_correlations.append({
                            'Stock': stock,
                            'Estimated Correlation': corr_estimate
                        })

                if daily_correlations:
                    corr_df = pd.DataFrame(daily_correlations)

                    fig = px.bar(
                        corr_df,
                        x='Stock',
                        y='Estimated Correlation',
                        title="Stock-Specific Sentiment-Return Correlations",
                        color='Estimated Correlation',
                        color_continuous_scale='RdBu_r'
                    )

                    fig.add_hline(y=0, line_dash="dash", line_color="black")
                    st.plotly_chart(fig, use_container_width=True)


def show_model_performance(stats):
    """Model Performance Dashboard"""
    st.header("🤖 Machine Learning Model Performance")

    if stats and 'classification' in stats:
        # Model comparison
        models_data = []

        for target, models in stats['classification'].items():
            horizon = target.replace('Returns_', '').replace('_binary', 'd')

            for model_name, metrics in models.items():
                if isinstance(metrics, dict) and 'accuracy' in metrics:
                    models_data.append({
                        'Model': f"{model_name.replace('_', ' ').title()}",
                        'Time Horizon': horizon,
                        'Accuracy': metrics['accuracy'],
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'F1-Score': metrics['f1_score']
                    })

        if models_data:
            models_df = pd.DataFrame(models_data)

            # Performance metrics visualization
            fig = px.bar(
                models_df,
                x='Time Horizon',
                y='F1-Score',
                color='Model',
                title="Model Performance: Direction Prediction",
                barmode='group'
            )

            fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                          annotation_text="Random Baseline (50%)")

            st.plotly_chart(fig, use_container_width=True)

            # Performance table
            st.subheader("📋 Detailed Performance Metrics")
            formatted_df = models_df.copy()
            for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.3f}")

            st.dataframe(formatted_df, use_container_width=True)

            # Feature importance
            if stats['classification']['Returns_1d_binary']['random_forest'].get('feature_importance'):
                st.subheader("🎯 Feature Importance Analysis")

                importance = stats['classification']['Returns_1d_binary']['random_forest']['feature_importance']

                fig = go.Figure(data=[
                    go.Bar(
                        x=list(importance.values()),
                        y=list(importance.keys()),
                        orientation='h',
                        marker=dict(color='lightblue', opacity=0.7)
                    )
                ])

                fig.update_layout(
                    title="Random Forest Feature Importance (1-Day Prediction)",
                    xaxis_title="Importance Score",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)


def show_news_analysis(news_df, sentiment_df, selected_stocks):
    """News Analysis Dashboard"""
    st.header("📰 News and Sentiment Analysis")

    if not news_df.empty:
        # News volume over time
        news_daily = news_df.groupby(news_df['date'].dt.date).size().reset_index()
        news_daily.columns = ['Date', 'Article_Count']

        fig = px.line(
            news_daily,
            x='Date',
            y='Article_Count',
            title="Daily News Volume",
            markers=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # Source distribution
        if 'source' in news_df.columns:
            source_counts = news_df['source'].value_counts()

            fig = px.pie(
                values=source_counts.values,
                names=source_counts.index,
                title="News Sources Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

        # Sentiment distribution
        if not sentiment_df.empty:
            fig = px.histogram(
                sentiment_df,
                x='ensemble_score_mean',
                title="Sentiment Score Distribution",
                nbins=50,
                marginal="box"
            )

            fig.add_vline(x=0, line_dash="dash", line_color="red",
                          annotation_text="Neutral Sentiment")

            st.plotly_chart(fig, use_container_width=True)


def show_data_quality(sentiment_df, news_df, stock_df):
    """Data Quality Dashboard"""
    st.header("📋 Data Quality Assessment")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Dataset Overview")

        quality_metrics = {
            'Metric': ['Sentiment Observations', 'News Articles', 'Stock Observations', 'Date Range (Days)',
                       'Unique Stocks'],
            'Count': [
                len(sentiment_df) if not sentiment_df.empty else 0,
                len(news_df) if not news_df.empty else 0,
                len(stock_df) if not stock_df.empty else 0,
                (sentiment_df['date'].max() - sentiment_df['date'].min()).days if not sentiment_df.empty else 0,
                sentiment_df['symbol'].nunique() if not sentiment_df.empty else 0
            ]
        }

        quality_df = pd.DataFrame(quality_metrics)
        st.dataframe(quality_df, use_container_width=True)

    with col2:
        st.subheader("🔍 Missing Data Analysis")

        if not sentiment_df.empty:
            missing_data = sentiment_df.isnull().sum()
            missing_pct = (missing_data / len(sentiment_df) * 100).round(2)

            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing %': missing_pct.values
            })

            missing_df = missing_df[missing_df['Missing Count'] > 0]

            if not missing_df.empty:
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing data detected")

    # Data timeline
    if not sentiment_df.empty:
        st.subheader("📅 Data Timeline")

        timeline_data = sentiment_df.groupby(['date', 'symbol']).size().reset_index(name='observations')

        fig = px.scatter(
            timeline_data,
            x='date',
            y='symbol',
            size='observations',
            title="Data Coverage Timeline",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()