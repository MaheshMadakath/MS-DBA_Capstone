import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="RQ2-RQ3 Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ff7f0e;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #f0f8ff;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .hypothesis-result {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: bold;
    }
    .hypothesis-accepted {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .hypothesis-rejected {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


def load_data():
    """Load all analysis results with error handling"""
    try:
        # Define data paths
        base_path = Path("rq2_rq3_analysis_results/dashboard_exports")

        data = {}

        # Load model performance results
        model_results_path = base_path / "model_performance_results.csv"
        if model_results_path.exists():
            data['model_results'] = pd.read_csv(model_results_path)
        else:
            st.error(f"Model results file not found: {model_results_path}")
            return None

        # Load feature importance
        feature_importance_path = base_path / "feature_importance_consensus.csv"
        if feature_importance_path.exists():
            data['feature_importance'] = pd.read_csv(feature_importance_path)
        else:
            st.warning("Feature importance file not found")
            data['feature_importance'] = pd.DataFrame()

        # Load hypothesis testing results
        hypothesis_path = base_path / "hypothesis_testing_results.json"
        if hypothesis_path.exists():
            with open(hypothesis_path, 'r') as f:
                data['hypothesis_results'] = json.load(f)
        else:
            st.warning("Hypothesis testing results not found")
            data['hypothesis_results'] = {}

        # Load analysis summary
        summary_path = base_path / "analysis_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                data['analysis_summary'] = json.load(f)
        else:
            st.warning("Analysis summary not found")
            data['analysis_summary'] = {}

        # Load SHAP importance if available
        shap_path = base_path / "shap_importance.csv"
        if shap_path.exists():
            data['shap_importance'] = pd.read_csv(shap_path)
        else:
            data['shap_importance'] = pd.DataFrame()

        return data

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def show_overall_summary(data):
    st.markdown('<div class="section-header">🧭 Overall Summary</div>', unsafe_allow_html=True)
    if 'model_results' not in data or data['model_results'].empty:
        st.info("No model results available for summary.")
        return

    df = data['model_results'].copy()

    # Pick a primary metric for ranking (F1 is balanced by default)
    metric = 'f1_score' if 'f1_score' in df.columns else df.filter(regex='(f1|accuracy|roc_auc)').columns[0]

    # Bundle summary
    bundle_means = df.groupby('bundle')[metric].agg(['mean','std','count']).sort_values('mean', ascending=False)
    best_bundle = bundle_means.index[0]
    best_bundle_mean = bundle_means.iloc[0]['mean']

    # Compare hybrid vs others if present
    compare_text = ""
    if 'hybrid' in bundle_means.index and 'technical_only' in bundle_means.index and 'sentiment_only' in bundle_means.index:
        d_hybrid_tech = best_bundle_mean - bundle_means.loc['technical_only','mean']
        d_hybrid_sent = best_bundle_mean - bundle_means.loc['sentiment_only','mean']
        compare_text = f"(Δ vs technical_only = {d_hybrid_tech:.3f}, Δ vs sentiment_only = {d_hybrid_sent:.3f})"

    # Model summary
    model_means = df.groupby('model')[metric].agg(['mean','std']).sort_values('mean', ascending=False)
    best_model = model_means.index[0]
    best_model_mean = model_means.iloc[0]['mean']

    # Horizon behavior (if present)
    horizon_text = ""
    if 'horizon' in df.columns:
        horizon_means = df.groupby('horizon')[metric].mean().sort_index()
        horizon_text = " • ".join([f"H{int(h)}: {v:.3f}" for h,v in horizon_means.items()])

    # Top features from the consensus file (if present)
    top_feats_text = ""
    if 'feature_importance' in data and not data['feature_importance'].empty:
        fi = data['feature_importance'].copy()
        rank_col = 'total_score' if 'total_score' in fi.columns else fi.filter(regex='(score|importance)').columns[0]
        top_feats = fi.sort_values(rank_col, ascending=False).head(5)['feature'].tolist()
        top_feats_text = ", ".join(top_feats)

    st.markdown(f"""
<div class="insight-box">
<strong>RQ2 – Best bundle:</strong> {best_bundle.replace('_',' ').title()} ({best_bundle_mean:.3f}) {compare_text}<br>
<strong>Best model:</strong> {best_model.replace('_',' ').title()} ({best_model_mean:.3f})<br>
<strong>Horizon pattern:</strong> {horizon_text if horizon_text else 'not available'}<br>
<strong>RQ3 – Top features (consensus):</strong> {top_feats_text if top_feats_text else 'not available'}
</div>
""", unsafe_allow_html=True)


def show_data_description(data):
    """Display comprehensive data description and EDA"""

    st.markdown('<div class="section-header">📊 Data Description & Exploratory Analysis</div>',
                unsafe_allow_html=True)

    if 'model_results' not in data or data['model_results'].empty:
        st.error("No model results data available")
        return

    df = data['model_results']

    # Dataset Overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>{}</h3>
            <p>Total Model Evaluations</p>
        </div>
        """.format(len(df)), unsafe_allow_html=True)

    with col2:
        unique_bundles = df['bundle'].nunique()
        st.markdown("""
        <div class="metric-card">
            <h3>{}</h3>
            <p>Feature Bundles</p>
        </div>
        """.format(unique_bundles), unsafe_allow_html=True)

    with col3:
        unique_models = df['model'].nunique()
        st.markdown("""
        <div class="metric-card">
            <h3>{}</h3>
            <p>ML Models</p>
        </div>
        """.format(unique_models), unsafe_allow_html=True)

    with col4:
        unique_horizons = df['horizon'].nunique()
        st.markdown("""
        <div class="metric-card">
            <h3>{}</h3>
            <p>Prediction Horizons</p>
        </div>
        """.format(unique_horizons), unsafe_allow_html=True)

    # Data Quality Assessment
    st.markdown("### 🔍 Data Quality Assessment")

    col1, col2 = st.columns(2)

    with col1:
        # Missing values analysis
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            fig_missing = px.bar(
                x=missing_data.index,
                y=missing_data.values,
                title="Missing Values by Column",
                labels={'x': 'Columns', 'y': 'Missing Count'}
            )
            fig_missing.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("✅ No missing values detected in the dataset")

    with col2:
        # Data distribution by bundle
        bundle_dist = df['bundle'].value_counts()
        fig_dist = px.pie(
            values=bundle_dist.values,
            names=bundle_dist.index,
            title="Evaluation Distribution by Feature Bundle"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Performance Metrics Distribution
    st.markdown("### 📈 Performance Metrics Distribution")

    # Why this analysis: Understanding the distribution of performance metrics helps identify
    # outliers, data quality issues, and gives insights into model behavior patterns

    metrics_cols = ['accuracy', 'f1_score', 'roc_auc', 'precision', 'recall']
    available_metrics = [col for col in metrics_cols if col in df.columns]

    if available_metrics:
        fig_metrics = make_subplots(
            rows=2, cols=3,
            subplot_titles=available_metrics,
            specs=[[{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}]]
        )

        for i, metric in enumerate(available_metrics[:6]):
            row = (i // 3) + 1
            col = (i % 3) + 1

            fig_metrics.add_trace(
                go.Histogram(x=df[metric], name=metric, showlegend=False),
                row=row, col=col
            )

        fig_metrics.update_layout(
            title_text="Distribution of Performance Metrics",
            height=600
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

        # Statistical Summary
        st.markdown("### 📊 Statistical Summary")
        summary_stats = df[available_metrics].describe()
        st.dataframe(summary_stats.round(4))

    # Feature Bundle Analysis
    st.markdown("### 🎯 Feature Bundle Characteristics")

    # Why this analysis: Different feature bundles represent different modeling approaches
    # Understanding their characteristics helps validate the research design

    if 'n_features' in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            # Feature count by bundle
            feature_count = df.groupby('bundle')['n_features'].mean().reset_index()
            fig_features = px.bar(
                feature_count,
                x='bundle',
                y='n_features',
                title="Average Number of Features by Bundle",
                color='bundle'
            )
            fig_features.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_features, use_container_width=True)

        with col2:
            # Sample size distribution
            sample_sizes = df.groupby('bundle')['test_size'].agg(['mean', 'std']).reset_index()
            fig_samples = px.bar(
                sample_sizes,
                x='bundle',
                y='mean',
                error_y='std',
                title="Average Test Sample Size by Bundle",
                color='bundle'
            )
            fig_samples.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_samples, use_container_width=True)

    # === Bundle Glossary (place inside show_data_description, after the feature-count bar) ===
    with st.expander("ℹ️ Bundle Glossary (exact features in each bundle)"):
        st.markdown("""
    **Sentiment bundles**
    - **sentiment_only (7):** ensemble_score_mean, ensemble_score_std, ensemble_score_count, confidence_mean, relevance_score_mean, sentiment_lag1, sentiment_lag2  
    - **sentiment_core (4):** ensemble_score_mean, ensemble_score_std, ensemble_score_count, confidence_mean

    **Technical bundles (subsets of 49 indicators)**
    - **technical_trend (15):** SMA_5, SMA_10, SMA_20, SMA_50, EMA_12, EMA_26, EMA_50, Price_vs_SMA20, Price_vs_SMA50, SMA_Ratio_5_20, SMA_Ratio_20_50, VOLUME_SMA_20, SMA5_above_SMA20, Price_above_SMA20, Price_above_SMA50
    - **technical_momentum (11):** RSI, MACD, MACD_Signal, MACD_Histogram, ROC_5, ROC_10, ROC_20, MOMENTUM_10, RSI_Oversold, RSI_Overbought, MACD_Bullish
    - **technical_volume (5):** OBV, VOLUME_SMA_20, VOLUME_RATIO, Volume_Price_Trend, Price_Volume_Correlation

    **All-technical & Hybrid**
    - **technical_only (49):** all 49 indicators listed in the analysis
    - **hybrid (56):** sentiment_only + all technical_only
    """)
        st.caption(
            "Note: Some indicators (e.g., VOLUME_SMA_20) appear in more than one specialized technical bundle by design.")

    # Temporal Analysis
    if 'horizon' in df.columns:
        st.markdown("### ⏰ Temporal Analysis (Prediction Horizons)")

        # Why this analysis: Different prediction horizons may show varying performance patterns
        # This helps understand model effectiveness across different time scales

        horizon_performance = df.groupby('horizon')[available_metrics].mean().reset_index()

        fig_temporal = px.line(
            horizon_performance.melt(id_vars=['horizon']),
            x='horizon',
            y='value',
            color='variable',
            title="Performance Across Prediction Horizons",
            labels={'variable': 'Metric', 'value': 'Performance Score'}
        )
        st.plotly_chart(fig_temporal, use_container_width=True)

        # Insight box
        st.markdown("""
        <div class="insight-box">
        <strong>💡 Insight:</strong> This temporal analysis reveals how model performance changes
        with prediction horizons. Generally, shorter horizons (1-day) tend to have higher accuracy
        due to stronger autocorrelation in financial time series, while longer horizons become
        more challenging due to increased uncertainty.
        </div>
        """, unsafe_allow_html=True)


def show_model_analysis(data):
    """Display comprehensive model performance analysis"""

    st.markdown('<div class="section-header">🤖 Model Performance Analysis</div>',
                unsafe_allow_html=True)

    if 'model_results' not in data or data['model_results'].empty:
        st.error("No model results data available")
        return

    df = data['model_results']

    # Model Selection Controls
    st.markdown("### 🎛️ Analysis Controls")
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_metric = st.selectbox(
            "Select Performance Metric",
            ['f1_score', 'accuracy', 'roc_auc', 'precision', 'recall'],
            help="Choose the primary metric for analysis. F1-score balances precision and recall."
        )

    with col2:
        selected_bundles = st.multiselect(
            "Select Feature Bundles",
            df['bundle'].unique(),
            default=df['bundle'].unique(),
            help="Compare different feature engineering approaches"
        )

    with col3:
        selected_models = st.multiselect(
            "Select Models",
            df['model'].unique(),
            default=df['model'].unique()[:4],  # Limit for readability
            help="Choose which ML models to compare"
        )

    # Filter data based on selections
    filtered_df = df[
        (df['bundle'].isin(selected_bundles)) &
        (df['model'].isin(selected_models))
        ]

    if filtered_df.empty:
        st.warning("No data available for selected filters")
        return

    # Performance Comparison by Bundle (Key for RQ2)
    st.markdown("### 🔬 RQ2: Feature Bundle Performance Comparison")

    # Why this analysis: This directly addresses RQ2 about which modeling approach performs better
    # Bundle comparison reveals whether hybrid approaches outperform individual approaches

    col1, col2 = st.columns(2)

    with col1:
        # Box plot for bundle comparison
        fig_bundle_box = px.box(
            filtered_df,
            x='bundle',
            y=selected_metric,
            title=f"{selected_metric.replace('_', ' ').title()} Distribution by Feature Bundle",
            color='bundle'
        )
        fig_bundle_box.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bundle_box, use_container_width=True)

    with col2:
        # Mean performance by bundle
        bundle_means = filtered_df.groupby('bundle')[selected_metric].agg(['mean', 'std']).reset_index()
        fig_bundle_bar = px.bar(
            bundle_means,
            x='bundle',
            y='mean',
            error_y='std',
            title=f"Mean {selected_metric.replace('_', ' ').title()} by Bundle",
            color='bundle'
        )
        fig_bundle_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bundle_bar, use_container_width=True)

    # Statistical Summary Table
    st.markdown("#### 📊 Bundle Performance Summary")
    bundle_summary = filtered_df.groupby('bundle')[selected_metric].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(4)

    # Add ranking
    bundle_summary = bundle_summary.sort_values('mean', ascending=False)
    bundle_summary['rank'] = range(1, len(bundle_summary) + 1)

    st.dataframe(bundle_summary)

    # Performance Heatmap
    st.markdown("### 🌡️ Model-Bundle Performance Heatmap")

    # Why this analysis: Reveals which model-bundle combinations work best
    # Helps identify optimal pairing strategies for deployment

    if len(selected_models) > 1 and len(selected_bundles) > 1:
        pivot_data = filtered_df.pivot_table(
            values=selected_metric,
            index='model',
            columns='bundle',
            aggfunc='mean'
        )

        fig_heatmap = px.imshow(
            pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            color_continuous_scale='RdYlBu_r',
            title=f"Model-Bundle Performance Heatmap ({selected_metric.replace('_', ' ').title()})",
            text_auto='.3f'
        )
        fig_heatmap.update_layout(height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # Model Comparison
    st.markdown("### ⚖️ Model Architecture Comparison")

    col1, col2 = st.columns(2)

    with col1:
        # Model performance ranking
        model_ranking = filtered_df.groupby('model')[selected_metric].agg(['mean', 'std']).sort_values('mean',
                                                                                                       ascending=False)

        fig_model_rank = px.bar(
            x=model_ranking.index,
            y=model_ranking['mean'],
            error_y=model_ranking['std'],
            title=f"Model Ranking by {selected_metric.replace('_', ' ').title()}",
            labels={'x': 'Model', 'y': f'{selected_metric.replace("_", " ").title()}'}
        )
        fig_model_rank.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_model_rank, use_container_width=True)

    with col2:
        # Model consistency (coefficient of variation)
        model_cv = filtered_df.groupby('model')[selected_metric].agg(['mean', 'std'])
        model_cv['cv'] = model_cv['std'] / model_cv['mean']
        model_cv = model_cv.sort_values('cv')

        fig_consistency = px.bar(
            x=model_cv.index,
            y=model_cv['cv'],
            title="Model Consistency (Lower = More Stable)",
            labels={'x': 'Model', 'y': 'Coefficient of Variation'}
        )
        fig_consistency.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_consistency, use_container_width=True)

    # Cross-Validation Results (if available)
    if 'cross_validation_results.csv' in [f.name for f in
                                          Path("rq2_rq3_analysis_results/dashboard_exports").glob("*.csv")]:
        st.markdown("### 🔄 Cross-Validation Stability Analysis")

        try:
            cv_path = Path("rq2_rq3_analysis_results/dashboard_exports/cross_validation_results.csv")
            cv_df = pd.read_csv(cv_path)

            # Why this analysis: Cross-validation results show model stability and generalization
            # Important for understanding which models are robust across different data splits

            if not cv_df.empty:
                cv_metrics = [col for col in cv_df.columns if col.endswith('_mean')]

                if cv_metrics:
                    fig_cv = px.bar(
                        cv_df,
                        x='model',
                        y=cv_metrics[0],  # Primary CV metric
                        error_y=cv_metrics[0].replace('_mean', '_std') if cv_metrics[0].replace('_mean',
                                                                                                '_std') in cv_df.columns else None,
                        title="Cross-Validation Performance",
                        color='model'
                    )
                    fig_cv.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_cv, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load cross-validation results: {e}")

    # Performance Insights
    st.markdown("### 💡 Key Performance Insights")

    # Automated insights based on analysis
    best_bundle = bundle_means.loc[bundle_means['mean'].idxmax(), 'bundle']
    best_model = model_ranking.index[0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="insight-box">
        <strong>🏆 Best Feature Bundle:</strong> {best_bundle.replace('_', ' ').title()}<br>
        <strong>Performance:</strong> {bundle_means.loc[bundle_means['bundle'] == best_bundle, 'mean'].iloc[0]:.4f}<br>
        <strong>Interpretation:</strong> This suggests that {
        'combining multiple feature types' if 'hybrid' in best_bundle
        else 'specialized feature engineering' if 'technical' in best_bundle or 'sentiment' in best_bundle
        else 'this approach'
        } yields the best predictive performance.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="insight-box">
        <strong>🎯 Best Model:</strong> {best_model.replace('_', ' ').title()}<br>
        <strong>Performance:</strong> {model_ranking.iloc[0]['mean']:.4f}<br>
        <strong>Stability:</strong> CV = {model_cv.loc[best_model, 'cv']:.3f}<br>
        <strong>Interpretation:</strong> This model shows the best balance of 
        performance and consistency across different feature combinations.
        </div>
        """, unsafe_allow_html=True)


def show_feature_analysis(data):
    """Display comprehensive feature importance analysis"""

    st.markdown('<div class="section-header">🎯 Feature Importance Analysis (RQ3)</div>',
                unsafe_allow_html=True)

    # Why this analysis: RQ3 focuses on identifying which KPIs/features matter most
    # This section provides multiple perspectives on feature importance for robust conclusions

    if 'feature_importance' not in data or data['feature_importance'].empty:
        st.warning("Feature importance data not available")
        return

    feature_df = data['feature_importance']

    # Feature Selection Controls
    st.markdown("### 🎛️ Feature Analysis Controls")
    col1, col2 = st.columns(2)

    with col1:
        top_n = st.slider(
            "Number of Top Features to Display",
            min_value=5, max_value=min(30, len(feature_df)),
            value=15,
            help="Select how many top features to analyze in detail"
        )

    with col2:
        sort_by = st.selectbox(
            "Sort Features By",
            ['total_score', 'selection_count', 'mean_score'],
            help="Choose ranking criteria: total_score (overall importance), selection_count (method consensus), mean_score (average ranking)"
        )

    # Sort and filter data
    feature_df_sorted = feature_df.sort_values(sort_by, ascending=False).head(top_n)

    # Top Features Visualization
    st.markdown("### 🏆 Top Performing Features")

    col1, col2 = st.columns(2)

    with col1:
        # Horizontal bar chart for feature importance
        fig_importance = px.bar(
            feature_df_sorted,
            x=sort_by,
            y='feature',
            orientation='h',
            title=f"Top {top_n} Features by {sort_by.replace('_', ' ').title()}",
            color=sort_by,
            color_continuous_scale='viridis'
        )
        fig_importance.update_layout(height=max(400, top_n * 25))
        st.plotly_chart(fig_importance, use_container_width=True)

    with col2:
        # Selection frequency analysis
        if 'selection_count' in feature_df.columns:
            selection_dist = feature_df['selection_count'].value_counts().sort_index()

            fig_selection = px.bar(
                x=selection_dist.index,
                y=selection_dist.values,
                title="Feature Selection Method Consensus",
                labels={'x': 'Number of Methods Selecting Feature', 'y': 'Number of Features'},
                color=selection_dist.values,
                color_continuous_scale='blues'
            )
            st.plotly_chart(fig_selection, use_container_width=True)

    # Feature Categories Analysis
    st.markdown("### 📊 Feature Categories Analysis")

    # Categorize features for better understanding
    def categorize_feature(feature_name):
        feature_lower = feature_name.lower()
        if any(term in feature_lower for term in ['sentiment', 'ensemble', 'confidence', 'relevance']):
            return 'Sentiment'
        elif any(term in feature_lower for term in ['rsi', 'macd', 'bb_', 'sma', 'ema']):
            return 'Technical Indicators'
        elif any(term in feature_lower for term in ['volume', 'obv']):
            return 'Volume'
        elif any(term in feature_lower for term in ['volatility', 'atr']):
            return 'Volatility'
        elif any(term in feature_lower for term in ['roc', 'momentum', 'price_position']):
            return 'Momentum'
        elif any(term in feature_lower for term in ['price_vs', 'above', 'oversold', 'overbought']):
            return 'Price Patterns'
        else:
            return 'Other'

    feature_df['category'] = feature_df['feature'].apply(categorize_feature)

    feature_df_sorted = feature_df.sort_values(sort_by, ascending=False).head(top_n)

    col1, col2 = st.columns(2)

    with col1:
        # Category distribution
        category_counts = feature_df['category'].value_counts()
        fig_categories = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Feature Distribution by Category"
        )
        st.plotly_chart(fig_categories, use_container_width=True)

    with col2:
        # Category performance
        category_performance = feature_df.groupby('category')[sort_by].agg(['mean', 'sum']).round(3)

        fig_cat_perf = px.bar(
            x=category_performance.index,
            y=category_performance[('mean')],
            title=f"Average {sort_by.replace('_', ' ').title()} by Category",
            color=category_performance[('mean')],
            color_continuous_scale='plasma'
        )
        fig_cat_perf.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_cat_perf, use_container_width=True)

    # SHAP Analysis (if available)
    if 'shap_importance' in data and not data['shap_importance'].empty:
        st.markdown("### 🔍 SHAP Feature Importance Analysis")

        # Why SHAP: Provides model-agnostic explanations showing how each feature
        # contributes to individual predictions, offering deeper insights than standard importance scores

        shap_df = data['shap_importance'].head(15)

        col1, col2 = st.columns(2)

        with col1:
            fig_shap = px.bar(
                shap_df,
                x='shap_importance',
                y='feature',
                orientation='h',
                title="SHAP Feature Importance (Top 15)",
                color='shap_importance',
                color_continuous_scale='reds'
            )
            fig_shap.update_layout(height=500)
            st.plotly_chart(fig_shap, use_container_width=True)

        with col2:
            # Compare SHAP with consensus ranking
            if 'total_score' in feature_df.columns:
                # Merge SHAP and consensus data
                merged_importance = pd.merge(
                    feature_df[['feature', 'total_score']],
                    shap_df[['feature', 'shap_importance']],
                    on='feature',
                    how='inner'
                )

                if not merged_importance.empty:
                    fig_comparison = px.scatter(
                        merged_importance,
                        x='total_score',
                        y='shap_importance',
                        text='feature',
                        title="SHAP vs Consensus Importance",
                        labels={
                            'total_score': 'Consensus Score',
                            'shap_importance': 'SHAP Importance'
                        }
                    )
                    fig_comparison.update_traces(textposition="top center")
                    st.plotly_chart(fig_comparison, use_container_width=True)

    # Feature Stability Analysis
    st.markdown("### 🎯 Feature Selection Stability")

    if 'selection_count' in feature_df.columns and 'mean_rank' in feature_df.columns:

        # Why stability matters: Features consistently selected across methods are more reliable
        # High stability indicates robust importance regardless of selection algorithm

        fig_stability = px.scatter(
            feature_df_sorted,
            x='selection_count',
            y='total_score',
            size='mean_rank',
            hover_name='feature',
            title="Feature Importance vs Selection Stability",
            labels={
                'selection_count': 'Number of Methods Selecting Feature',
                'total_score': 'Total Importance Score',
                'mean_rank': 'Average Rank (Size)'
            },
            color='total_score',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_stability, use_container_width=True)

        # Identify highly stable features
        stable_features = feature_df[feature_df['selection_count'] >= feature_df['selection_count'].quantile(0.8)]

        if not stable_features.empty:
            st.markdown("#### 🛡️ Most Stable Features")
            st.markdown("""
            <div class="insight-box">
            <strong>Highly Stable Features (Selected by ≥80% of methods):</strong><br>
            """ + ", ".join(stable_features.head(10)['feature'].tolist()) + """<br><br>
            <strong>Why this matters:</strong> These features show consistent importance across 
            different selection algorithms, indicating robust predictive value regardless of the 
            feature engineering approach used.
            </div>
            """, unsafe_allow_html=True)

    # Feature Rankings Table
    st.markdown("### 📋 Detailed Feature Rankings")

    # Display comprehensive feature table
    display_cols = ['feature', 'total_score', 'selection_count', 'mean_rank', 'category']
    available_cols = [col for col in display_cols if col in feature_df_sorted.columns]

    if available_cols:
        st.dataframe(
            feature_df_sorted[available_cols].round(4),
            use_container_width=True
        )


def show_hypothesis_results(data):
    """Display hypothesis testing results for RQ2 and RQ3"""

    st.markdown('<div class="section-header">🔬 Hypothesis Testing Results</div>',
                unsafe_allow_html=True)

    # Why hypothesis testing: Provides statistical validation for research conclusions
    # Essential for academic/scientific rigor and business decision confidence

    if 'hypothesis_results' not in data or not data['hypothesis_results']:
        st.warning("Hypothesis testing results not available")
        return

    hypothesis_data = data['hypothesis_results']

    # Research Questions Overview
    st.markdown("### 📋 Research Questions & Hypotheses")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **RQ2: Technical vs Sentiment Comparison**
        - **H0**: Hybrid models ≠ individual models  
        - **H1**: Hybrid > individual models
        - **Purpose**: Strategy Optimization - Prevents suboptimal model deployment
        """)

    with col2:
        st.markdown("""
        **RQ3: Feature Importance Analysis**
        - **H0**: No KPI dominates
        - **H1**: Some KPIs stronger
        - **Purpose**: Feature Prioritization - Focus on high-impact signals
        """)

    # RQ2 Results
    if 'rq2_analysis' in hypothesis_data:
        st.markdown("### 🎯 RQ2: Hybrid vs Individual Models Analysis")

        rq2_data = hypothesis_data['rq2_analysis']
        rq2_tests = rq2_data.get('statistical_tests', [])
        rq2_summary = rq2_data.get('summary', {})

        # Overall conclusion
        conclusion = rq2_summary.get('overall_conclusion', 'No conclusion available')
        h1_supported = 'support for h1' in conclusion.lower() or 'accept h1' in conclusion.lower()

        conclusion_class = 'hypothesis-accepted' if h1_supported else 'hypothesis-rejected'
        conclusion_icon = '✅' if h1_supported else '❌'

        st.markdown(f"""
        <div class="hypothesis-result {conclusion_class}">
        {conclusion_icon} <strong>RQ2 Result:</strong> {conclusion}
        </div>
        """, unsafe_allow_html=True)

        # Statistical test details
        if rq2_tests:
            st.markdown("#### 📊 Statistical Test Results")

            # Filter for main hypothesis tests
            main_tests = [t for t in rq2_tests if t.get('comparison') == 'hybrid_vs_individual']

            if main_tests:
                test_results = []
                for test in main_tests:
                    test_result = test.get('test_result', {})
                    test_results.append({
                        'Metric': test.get('metric', '').replace('_', ' ').title(),
                        'Hybrid Mean': f"{test.get('hybrid_mean', 0):.4f}",
                        'Individual Mean': f"{test.get('individual_mean', 0):.4f}",
                        'T-Statistic': f"{test_result.get('test_statistic', 0):.4f}",
                        'P-Value': f"{test_result.get('p_value', 1):.4f}",
                        'Effect Size': f"{test_result.get('effect_size', 0):.4f}",
                        'Significant': '✅' if test_result.get('significant', False) else '❌'
                    })

                if test_results:
                    st.dataframe(pd.DataFrame(test_results), use_container_width=True)

            # Visualization of p-values
            p_values = []
            metrics = []
            significance = []

            for test in main_tests:
                test_result = test.get('test_result', {})
                p_values.append(test_result.get('p_value', 1))
                metrics.append(test.get('metric', '').replace('_', ' ').title())
                significance.append(test_result.get('significant', False))

            if p_values:
                fig_pvals = px.bar(
                    x=metrics,
                    y=[-np.log10(p) for p in p_values],
                    title="Statistical Significance (-log10 p-value)",
                    labels={'x': 'Performance Metric', 'y': '-log10(p-value)'},
                    color=['Significant' if sig else 'Not Significant' for sig in significance],
                    color_discrete_map={'Significant': 'green', 'Not Significant': 'red'}
                )
                fig_pvals.add_hline(y=-np.log10(0.05), line_dash="dash",
                                    annotation_text="α = 0.05", annotation_position="top right")
                st.plotly_chart(fig_pvals, use_container_width=True)

        # Summary statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            total_tests = rq2_summary.get('total_tests_performed', 0)
            st.metric("Total Tests", total_tests)

        with col2:
            significant_tests = rq2_summary.get('significant_tests', 0)
            st.metric("Significant Results", significant_tests)

        with col3:
            h1_support = rq2_summary.get('h1_support_tests', 0)
            st.metric("H1 Support Count", h1_support)

    # RQ3 Results
    if 'rq3_analysis' in hypothesis_data:
        st.markdown("### 🎯 RQ3: Feature Importance Dominance Analysis")

        rq3_data = hypothesis_data['rq3_analysis']
        rq3_tests = rq3_data.get('statistical_tests', [])
        rq3_summary = rq3_data.get('summary', {})

        # Overall conclusion
        conclusion = rq3_summary.get('overall_conclusion', 'No conclusion available')
        h1_supported = 'support for h1' in conclusion.lower() or 'accept h1' in conclusion.lower()

        conclusion_class = 'hypothesis-accepted' if h1_supported else 'hypothesis-rejected'
        conclusion_icon = '✅' if h1_supported else '❌'

        st.markdown(f"""
        <div class="hypothesis-result {conclusion_class}">
        {conclusion_icon} <strong>RQ3 Result:</strong> {conclusion}
        </div>
        """, unsafe_allow_html=True)

        # Feature dominance details
        if rq3_tests:
            st.markdown("#### 🔍 Feature Dominance Tests")

            test_results = []
            for test in rq3_tests:
                test_results.append({
                    'Test Name': test.get('test_type', '').replace('_', ' ').title(),
                    'Test Statistic': f"{test.get('test_result', {}).get('test_statistic', 0):.4f}",
                    'P-Value': f"{test.get('test_result', {}).get('p_value', 1):.4f}",
                    'Significant': '✅' if test.get('test_result', {}).get('significant', False) else '❌',
                    'Interpretation': test.get('test_result', {}).get('conclusion', 'No interpretation')
                })

            if test_results:
                st.dataframe(pd.DataFrame(test_results), use_container_width=True)

        # Top feature information
        top_feature = rq3_summary.get('most_important_feature', 'N/A')
        if top_feature != 'N/A':
            st.markdown(f"""
            <div class="insight-box">
            <strong>🏆 Most Important Feature:</strong> {top_feature}<br>
            <strong>Score:</strong> {rq3_summary.get('top_feature_score', 0):.4f}<br>
            <strong>Impact:</strong> This feature emerged as the strongest predictor across multiple selection methods.
            </div>
            """, unsafe_allow_html=True)

    # Overall Research Implications
    st.markdown("### 💼 Business & Research Implications")

    implications = hypothesis_data.get('overall_summary', {}).get('research_implications', [])

    if implications:
        for i, implication in enumerate(implications, 1):
            st.markdown(f"{i}. {implication}")
    else:
        # Default implications based on available data
        rq2_conclusion = hypothesis_data.get('rq2_analysis', {}).get('summary', {}).get('overall_conclusion', '')
        rq3_conclusion = hypothesis_data.get('rq3_analysis', {}).get('summary', {}).get('overall_conclusion', '')

        implications_text = "### Key Implications:\n\n"

        if 'support' in rq2_conclusion.lower():
            implications_text += "- **Model Strategy**: Hybrid approaches recommended for optimal performance\n"
            implications_text += "- **Resource Allocation**: Invest in multi-modal feature engineering\n"
        else:
            implications_text += "- **Model Strategy**: Individual specialized models may be sufficient\n"
            implications_text += "- **Resource Allocation**: Focus on domain-specific feature engineering\n"

        if 'support' in rq3_conclusion.lower():
            implications_text += "- **Feature Focus**: Prioritize identified dominant features\n"
            implications_text += "- **Development**: Concentrate on high-impact indicators\n"
        else:
            implications_text += "- **Feature Focus**: Maintain broad feature coverage\n"
            implications_text += "- **Development**: Continue comprehensive signal processing\n"

        st.markdown(implications_text)


def show_additional_analysis(data):
    """Display additional analysis and insights"""

    st.markdown('<div class="section-header">🔍 Additional Analysis & Insights</div>',
                unsafe_allow_html=True)

    # Model Efficiency Analysis
    st.markdown("### ⚡ Model Efficiency Analysis")

    if 'model_results' in data and not data['model_results'].empty:
        df = data['model_results']

        # Feature efficiency (performance per feature)
        if 'n_features' in df.columns and 'f1_score' in df.columns:
            df['efficiency'] = df['f1_score'] / df['n_features']

            col1, col2 = st.columns(2)

            with col1:
                # Efficiency by bundle
                efficiency_by_bundle = df.groupby('bundle')['efficiency'].mean().sort_values(ascending=False)

                fig_efficiency = px.bar(
                    x=efficiency_by_bundle.index,
                    y=efficiency_by_bundle.values,
                    title="Model Efficiency (Performance per Feature)",
                    labels={'x': 'Feature Bundle', 'y': 'F1-Score per Feature'},
                    color=efficiency_by_bundle.values,
                    color_continuous_scale='viridis'
                )
                fig_efficiency.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_efficiency, use_container_width=True)

            with col2:
                # Efficiency vs complexity scatter
                fig_scatter = px.scatter(
                    df,
                    x='n_features',
                    y='f1_score',
                    color='bundle',
                    size='efficiency',
                    title="Performance vs Complexity",
                    labels={'n_features': 'Number of Features', 'f1_score': 'F1-Score'},
                    hover_data=['model']
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("""
            <div class="insight-box">
            <strong>💡 Efficiency Insight:</strong> This analysis reveals which approaches provide 
            the best performance-to-complexity ratio. Higher efficiency indicates better performance 
            with fewer features, suggesting more parsimonious and interpretable models.
            </div>
            """, unsafe_allow_html=True)

    # Consistency Analysis
    st.markdown("### 🎯 Model Consistency Analysis")

    if 'model_results' in data:
        df = data['model_results']

        # Calculate coefficient of variation for each model-bundle combination
        consistency_df = df.groupby(['model', 'bundle'])['f1_score'].agg(['mean', 'std']).reset_index()
        consistency_df['cv'] = consistency_df['std'] / consistency_df['mean']
        consistency_df = consistency_df.dropna()

        if not consistency_df.empty:
            # Pivot for heatmap
            consistency_pivot = consistency_df.pivot(index='model', columns='bundle', values='cv')

            fig_consistency_heatmap = px.imshow(
                consistency_pivot.values,
                x=consistency_pivot.columns,
                y=consistency_pivot.index,
                color_continuous_scale='RdYlBu',
                title="Model Consistency (Lower = More Consistent)",
                text_auto='.3f'
            )
            fig_consistency_heatmap.update_layout(height=400)
            st.plotly_chart(fig_consistency_heatmap, use_container_width=True)

            # Most consistent models
            most_consistent = consistency_df.nsmallest(5, 'cv')[['model', 'bundle', 'mean', 'cv']]

            st.markdown("#### 🛡️ Most Consistent Model-Bundle Combinations")
            st.dataframe(
                most_consistent.round(4).rename(columns={
                    'mean': 'Mean F1-Score',
                    'cv': 'Coefficient of Variation'
                }),
                use_container_width=True
            )

    # Risk-Return Analysis
    st.markdown("### 📊 Risk-Return Analysis")

    # Why this analysis: In financial modeling, we need to balance performance (return) with
    # stability (risk). Models with high variance might not be suitable for production deployment.

    if 'model_results' in data and len(data['model_results']) > 0:
        df = data['model_results']

        # Calculate risk-return metrics by model
        risk_return = df.groupby('model').agg({
            'f1_score': ['mean', 'std'],
            'accuracy': ['mean', 'std'],
            'roc_auc': ['mean', 'std']
        }).round(4)

        # Flatten column names
        risk_return.columns = ['_'.join(col).strip() for col in risk_return.columns]
        risk_return = risk_return.reset_index()

        # Sharpe-like ratio (return/risk)
        risk_return['f1_sharpe'] = risk_return['f1_score_mean'] / (risk_return['f1_score_std'] + 1e-8)

        col1, col2 = st.columns(2)

        with col1:
            # Risk-Return scatter
            fig_risk_return = px.scatter(
                risk_return,
                x='f1_score_std',
                y='f1_score_mean',
                text='model',
                title="Risk-Return Profile (F1-Score)",
                labels={'f1_score_std': 'Risk (Standard Deviation)', 'f1_score_mean': 'Return (Mean F1-Score)'},
                color='f1_sharpe',
                color_continuous_scale='viridis'
            )
            fig_risk_return.update_traces(textposition="top center")
            st.plotly_chart(fig_risk_return, use_container_width=True)

        with col2:
            # Sharpe ratio ranking
            sharpe_ranking = risk_return.nlargest(8, 'f1_sharpe')[
                ['model', 'f1_sharpe', 'f1_score_mean', 'f1_score_std']]

            fig_sharpe = px.bar(
                sharpe_ranking,
                x='model',
                y='f1_sharpe',
                title="Risk-Adjusted Performance Ranking",
                labels={'f1_sharpe': 'Risk-Adjusted Score', 'model': 'Model'},
                color='f1_sharpe',
                color_continuous_scale='plasma'
            )
            fig_sharpe.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_sharpe, use_container_width=True)

    # Feature Category Deep Dive
    if 'feature_importance' in data and not data['feature_importance'].empty:
        st.markdown("### 🔬 Feature Category Deep Dive")

        feature_df = data['feature_importance']

        # Advanced feature categorization
        def advanced_categorize(feature_name):
            feature_lower = feature_name.lower()
            categories = []

            if any(term in feature_lower for term in ['sentiment', 'ensemble', 'confidence']):
                categories.append('Sentiment')
            if any(term in feature_lower for term in ['rsi', 'macd']):
                categories.append('Momentum Oscillators')
            if any(term in feature_lower for term in ['sma', 'ema', 'bb_']):
                categories.append('Trend Indicators')
            if any(term in feature_lower for term in ['volume', 'obv']):
                categories.append('Volume')
            if any(term in feature_lower for term in ['volatility', 'atr']):
                categories.append('Volatility')
            if any(term in feature_lower for term in ['price_vs', 'above', 'position']):
                categories.append('Price Patterns')

            return categories[0] if categories else 'Other'

        feature_df['detailed_category'] = feature_df['feature'].apply(advanced_categorize)

        # Category performance analysis
        category_analysis = feature_df.groupby('detailed_category').agg({
            'total_score': ['count', 'mean', 'sum'],
            'selection_count': 'mean'
        }).round(3)

        category_analysis.columns = ['Feature_Count', 'Avg_Score', 'Total_Score', 'Avg_Selection_Count']
        category_analysis = category_analysis.reset_index()

        # Bubble chart for category analysis
        fig_category_bubble = px.scatter(
            category_analysis,
            x='Avg_Score',
            y='Avg_Selection_Count',
            size='Feature_Count',
            color='Total_Score',
            text='detailed_category',
            title="Feature Category Performance Analysis",
            labels={
                'Avg_Score': 'Average Importance Score',
                'Avg_Selection_Count': 'Average Selection Frequency',
                'Feature_Count': 'Number of Features (Size)',
                'Total_Score': 'Total Category Score (Color)'
            }
        )
        fig_category_bubble.update_traces(textposition="middle center")
        st.plotly_chart(fig_category_bubble, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>📈 Category Analysis Insight:</strong> This visualization shows which feature categories 
        are most valuable overall. Position (avg score vs selection frequency) indicates reliability, 
        size shows category breadth, and color shows total impact. Categories in the top-right with 
        large bubbles are most valuable for model development.
        </div>
        """, unsafe_allow_html=True)


# Main Dashboard
def main():
    """Main dashboard function"""

    # Header
    st.markdown('<div class="main-header">RQ2-RQ3 Analysis Dashboard</div>',
                unsafe_allow_html=True)

    st.markdown("""
    ### Research Questions:
    - **RQ2**: Technical vs Sentiment Comparison - Which models perform better?
    - **RQ3**: Feature Importance Analysis - Which KPIs matter most?
    """)

    # Load data
    with st.spinner("Loading analysis results..."):
        data = load_data()

    if data is None:
        st.error("Failed to load analysis data. Please ensure the analysis has been completed.")
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Analysis Section",
        [
            "📊 Summary",
            "📊 Data Description & EDA",
            "🤖 Model Performance Analysis",
            "🎯 Feature Importance Analysis",
            "🔬 Hypothesis Testing Results",
            "🔍 Additional Analysis"
        ]
    )

    # Display selected section
    if page == "📊 Summary":
        show_overall_summary(data)
    elif page == "📊 Data Description & EDA":
        show_data_description(data)
    elif page == "🤖 Model Performance Analysis":
        show_model_analysis(data)
    elif page == "🎯 Feature Importance Analysis":
        show_feature_analysis(data)
    elif page == "🔬 Hypothesis Testing Results":
        show_hypothesis_results(data)
    elif page == "🔍 Additional Analysis":
        show_additional_analysis(data)

    # Footer
    st.markdown("---")
    st.markdown("""
    **Analysis Framework**: RQ2-RQ3 Comprehensive Analysis  
    **Statistical Methods**: Hypothesis testing with effect size analysis  
    **Model Validation**: Time-series cross-validation with multiple metrics
    """)


if __name__ == "__main__":
    main()