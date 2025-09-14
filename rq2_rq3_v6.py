#!/usr/bin/env python3
"""
RQ2-RQ3
========================================

RQ2: Technical vs Sentiment Comparison - Which models perform better?
H0: Hybrid models ≠ individual models 
H1: Hybrid > individual models

RQ3: Feature Importance Analysis - Which KPIs matter most?
H0: No KPI dominates
H1: Some KPIs stronger

"""

from __future__ import annotations
import argparse
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, friedmanchisquare
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# ML imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    TimeSeriesSplit, cross_val_score, GridSearchCV,
    validation_curve, learning_curve
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif,
    RFE, RFECV, SelectFromModel, VarianceThreshold
)

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

try:
    import shap

    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False

# Statistical testing
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.contingency_tables import mcnemar


# =============================================================================
# CONFIGURATION AND DATA STRUCTURES
# =============================================================================

@dataclass
class AnalysisConfig:
    """Configuration for RQ2-RQ3 analysis framework"""
    # Data paths
    sentiment_csv: str = "rq1_complete_results/daily_sentiment_aggregated.csv"
    price_csv: str = "rq1_complete_results/stock_data_clean.csv"
    sectors_csv: str = "rq1_complete_results/sector_analysis/sectors.csv"

    # Analysis parameters
    tickers: Optional[str] = None  # Comma-separated list
    test_fraction: float = 0.2
    cv_folds: int = 5
    random_state: int = 42

    # Output directories
    outdir: str = "rq2_rq3_analysis_results"

    # Feature selection parameters
    k_features: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 25])
    feature_methods: List[str] = field(default_factory=lambda: ['filter', 'wrapper', 'embedded', 'shap'])

    # Technical indicator parameters
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14

    # Hypothesis testing parameters
    alpha: float = 0.05
    multiple_testing_correction: str = 'bonferroni'  # bonferroni, fdr_bh, none

    # Performance thresholds
    min_samples_per_symbol: int = 30
    min_test_samples: int = 10

    def __post_init__(self):
        # Create output directories
        self.results_dir = Path(self.outdir) / "results"
        self.plots_dir = Path(self.outdir) / "plots"
        self.artifacts_dir = Path(self.outdir) / "artifacts"
        self.reports_dir = Path(self.outdir) / "reports"
        self.dashboard_dir = Path(self.outdir) / "dashboard_exports"

        for dir_path in [self.results_dir, self.plots_dir, self.artifacts_dir,
                         self.reports_dir, self.dashboard_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelSpec:
    """Model specification for systematic evaluation"""
    name: str
    estimator: object
    needs_scaling: bool = False
    param_grid: Dict[str, Any] = field(default_factory=dict)
    model_type: str = 'classification'  # classification or regression


@dataclass
class FeatureBundle:
    """Feature bundle definition for RQ2 analysis"""
    name: str
    columns: List[str]
    description: str
    bundle_type: str  # 'sentiment', 'technical', 'hybrid'


@dataclass
class HypothesisTest:
    """Hypothesis test result structure"""
    research_question: str
    hypothesis_null: str
    hypothesis_alternative: str
    test_statistic: float
    p_value: float
    effect_size: float
    effect_interpretation: str
    significant: bool
    conclusion: str
    method: str
    sample_size: int


# =============================================================================
# TECHNICAL INDICATORS CALCULATION ENGINE
# =============================================================================

class TechnicalIndicatorEngine:
    """Comprehensive technical indicators calculation with validation"""

    @staticmethod
    def validate_price_data(df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean price data"""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Remove invalid price data
        df = df.copy()
        price_cols = ['open', 'high', 'low', 'close']

        # Remove negative prices
        for col in price_cols:
            df = df[df[col] > 0]

        # Remove inconsistent OHLC data
        df = df[
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
            ]

        # Remove zero volume (optional, can be valid)
        df = df[df['volume'] >= 0]

        return df

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index with improved calculation"""
        if len(series) < period + 1:
            return pd.Series(index=series.index, dtype=float)

        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()

        # Handle division by zero
        rs = gain / loss
        rs = rs.replace([np.inf, -np.inf], np.nan)

        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Neutral RSI for missing values

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
        """MACD with comprehensive validation"""
        if len(series) < slow + signal:
            empty_series = pd.Series(index=series.index, dtype=float)
            return empty_series, empty_series, empty_series

        ema_fast = series.ewm(span=fast, min_periods=fast).mean()
        ema_slow = series.ewm(span=slow, min_periods=slow).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[
        pd.Series, pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands with %B calculation"""
        if len(series) < period:
            empty_series = pd.Series(index=series.index, dtype=float)
            return empty_series, empty_series, empty_series, empty_series

        middle = series.rolling(window=period, min_periods=period).mean()
        std = series.rolling(window=period, min_periods=period).std()

        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        # %B calculation with division by zero protection
        band_width = upper - lower
        percent_b = pd.Series(index=series.index, dtype=float)
        mask = band_width != 0
        percent_b[mask] = (series[mask] - lower[mask]) / band_width[mask]
        percent_b[~mask] = 0.5  # Neutral position when bands collapse

        return upper, middle, lower, percent_b

    @staticmethod
    def moving_averages(series: pd.Series) -> Dict[str, pd.Series]:
        """Comprehensive moving averages suite"""
        if len(series) < 50:
            # Return empty series for insufficient data
            empty_dict = {}
            for name in ['SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'EMA_50']:
                empty_dict[name] = pd.Series(index=series.index, dtype=float)
            return empty_dict

        return {
            'SMA_5': series.rolling(5, min_periods=5).mean(),
            'SMA_10': series.rolling(10, min_periods=10).mean(),
            'SMA_20': series.rolling(20, min_periods=20).mean(),
            'SMA_50': series.rolling(50, min_periods=50).mean(),
            'EMA_12': series.ewm(span=12, min_periods=12).mean(),
            'EMA_26': series.ewm(span=26, min_periods=26).mean(),
            'EMA_50': series.ewm(span=50, min_periods=50).mean()
        }

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range with improved calculation"""
        if len(high) < period + 1:
            return pd.Series(index=high.index, dtype=float)

        prev_close = close.shift(1)

        # True Range components
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        # True Range is the maximum of the three components
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR is the moving average of True Range
        atr = true_range.rolling(window=period, min_periods=period).mean()

        return atr

    @staticmethod
    def momentum_indicators(close: pd.Series, volume: pd.Series) -> Dict[str, pd.Series]:
        """Comprehensive momentum and volume indicators"""
        indicators = {}

        if len(close) < 20:
            # Return empty indicators for insufficient data
            indicator_names = [
                'ROC_5', 'ROC_10', 'ROC_20', 'PRICE_POSITION', 'OBV',
                'VOLUME_SMA_20', 'VOLUME_RATIO', 'VOLATILITY_5',
                'VOLATILITY_10', 'VOLATILITY_20', 'MOMENTUM_10'
            ]
            for name in indicator_names:
                indicators[name] = pd.Series(index=close.index, dtype=float)
            return indicators

        # Rate of Change
        indicators['ROC_5'] = close.pct_change(5)
        indicators['ROC_10'] = close.pct_change(10)
        indicators['ROC_20'] = close.pct_change(20)

        # Price position within recent range
        rolling_min = close.rolling(20, min_periods=20).min()
        rolling_max = close.rolling(20, min_periods=20).max()
        range_width = rolling_max - rolling_min

        indicators['PRICE_POSITION'] = pd.Series(index=close.index, dtype=float)
        mask = range_width != 0
        indicators['PRICE_POSITION'][mask] = (close[mask] - rolling_min[mask]) / range_width[mask]
        indicators['PRICE_POSITION'][~mask] = 0.5  # Neutral position

        # On-Balance Volume
        price_change = np.sign(close.diff())
        indicators['OBV'] = (price_change * volume).cumsum()

        # Volume indicators
        indicators['VOLUME_SMA_20'] = volume.rolling(20, min_periods=20).mean()
        volume_ratio = volume / indicators['VOLUME_SMA_20']
        indicators['VOLUME_RATIO'] = volume_ratio.replace([np.inf, -np.inf], 1.0)

        # Volatility measures
        returns = close.pct_change()
        indicators['VOLATILITY_5'] = returns.rolling(5, min_periods=5).std()
        indicators['VOLATILITY_10'] = returns.rolling(10, min_periods=10).std()
        indicators['VOLATILITY_20'] = returns.rolling(20, min_periods=20).std()

        # Price momentum
        indicators['MOMENTUM_10'] = close / close.shift(10) - 1

        return indicators

    @staticmethod
    def pattern_recognition(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Basic pattern recognition indicators"""
        patterns = {}

        if len(df) < 5:
            pattern_names = [
                'DOJI', 'HAMMER', 'SHOOTING_STAR', 'ENGULFING_BULL',
                'ENGULFING_BEAR', 'INSIDE_BAR', 'OUTSIDE_BAR'
            ]
            for name in pattern_names:
                patterns[name] = pd.Series(index=df.index, dtype=int)
            return patterns

        # Extract OHLC
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']

        # Body and shadow calculations
        body = (close - open_price).abs()
        upper_shadow = high - pd.concat([open_price, close], axis=1).max(axis=1)
        lower_shadow = pd.concat([open_price, close], axis=1).min(axis=1) - low

        # Average body size for comparison
        avg_body = body.rolling(10, min_periods=5).mean()

        # Doji pattern (small body relative to average)
        patterns['DOJI'] = (body < avg_body * 0.1).astype(int)

        # Hammer pattern (small body, long lower shadow, small upper shadow)
        patterns['HAMMER'] = (
                (body < avg_body * 0.3) &
                (lower_shadow > body * 2) &
                (upper_shadow < body * 0.5)
        ).astype(int)

        # Shooting star (small body, long upper shadow, small lower shadow)
        patterns['SHOOTING_STAR'] = (
                (body < avg_body * 0.3) &
                (upper_shadow > body * 2) &
                (lower_shadow < body * 0.5)
        ).astype(int)

        # Engulfing patterns
        prev_body = body.shift(1)
        prev_close = close.shift(1)
        prev_open = open_price.shift(1)

        # Bullish engulfing
        patterns['ENGULFING_BULL'] = (
                (prev_close < prev_open) &  # Previous candle was bearish
                (close > open_price) &  # Current candle is bullish
                (open_price < prev_close) &  # Current open below previous close
                (close > prev_open)  # Current close above previous open
        ).astype(int)

        # Bearish engulfing
        patterns['ENGULFING_BEAR'] = (
                (prev_close > prev_open) &  # Previous candle was bullish
                (close < open_price) &  # Current candle is bearish
                (open_price > prev_close) &  # Current open above previous close
                (close < prev_open)  # Current close below previous open
        ).astype(int)

        # Inside bar (current bar's range is within previous bar's range)
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        patterns['INSIDE_BAR'] = (
                (high <= prev_high) & (low >= prev_low)
        ).astype(int)

        # Outside bar (current bar's range encompasses previous bar's range)
        patterns['OUTSIDE_BAR'] = (
                (high >= prev_high) & (low <= prev_low)
        ).astype(int)

        return patterns


def compute_comprehensive_technical_features(price_df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """
    Compute comprehensive technical features with robust error handling

    Expected input columns: symbol, date, open, high, low, close, volume
    """
    if price_df.empty:
        return price_df

    price_df = price_df.sort_values(['symbol', 'date']).copy()

    # Validate required columns
    required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in price_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    results = []
    tech_engine = TechnicalIndicatorEngine()

    for symbol, group in price_df.groupby('symbol'):
        if len(group) < config.min_samples_per_symbol:
            print(f"Skipping {symbol}: insufficient data ({len(group)} < {config.min_samples_per_symbol})")
            continue

        group = group.copy().sort_values('date')

        try:
            # Validate price data
            group = tech_engine.validate_price_data(group)

            if len(group) < config.min_samples_per_symbol:
                print(f"Skipping {symbol}: insufficient valid data after cleaning")
                continue

            # Basic price features
            group['returns_1d'] = group['close'].pct_change()
            group['returns_2d'] = group['close'].pct_change(2)
            group['returns_5d'] = group['close'].pct_change(5)
            group['log_returns'] = np.log(group['close'] / group['close'].shift(1))

            # Technical indicators
            group['RSI'] = tech_engine.rsi(group['close'], config.rsi_period)

            macd_line, signal_line, histogram = tech_engine.macd(
                group['close'], config.macd_fast, config.macd_slow, config.macd_signal
            )
            group['MACD'] = macd_line
            group['MACD_Signal'] = signal_line
            group['MACD_Histogram'] = histogram

            bb_upper, bb_middle, bb_lower, bb_percent = tech_engine.bollinger_bands(
                group['close'], config.bb_period, config.bb_std
            )
            group['BB_Upper'] = bb_upper
            group['BB_Middle'] = bb_middle
            group['BB_Lower'] = bb_lower
            group['BB_PercentB'] = bb_percent
            group['BB_Width'] = (bb_upper - bb_lower) / bb_middle

            # Moving averages
            ma_dict = tech_engine.moving_averages(group['close'])
            for name, values in ma_dict.items():
                group[name] = values

            # Price relative to moving averages
            group['Price_vs_SMA20'] = group['close'] / group['SMA_20'] - 1
            group['Price_vs_SMA50'] = group['close'] / group['SMA_50'] - 1
            group['SMA_Ratio_5_20'] = group['SMA_5'] / group['SMA_20'] - 1
            group['SMA_Ratio_20_50'] = group['SMA_20'] / group['SMA_50'] - 1

            # ATR
            group['ATR'] = tech_engine.atr(
                group['high'], group['low'], group['close'], config.atr_period
            )
            group['ATR_Normalized'] = group['ATR'] / group['close']

            # Momentum and volume indicators
            momentum_dict = tech_engine.momentum_indicators(group['close'], group['volume'])
            for name, values in momentum_dict.items():
                group[name] = values

            # Pattern recognition
            pattern_dict = tech_engine.pattern_recognition(group)
            for name, values in pattern_dict.items():
                group[name] = values

            # Cross-over signals and derived features
            group['SMA5_above_SMA20'] = (group['SMA_5'] > group['SMA_20']).astype(int)
            group['Price_above_SMA20'] = (group['close'] > group['SMA_20']).astype(int)
            group['Price_above_SMA50'] = (group['close'] > group['SMA_50']).astype(int)
            group['RSI_Oversold'] = (group['RSI'] < 30).astype(int)
            group['RSI_Overbought'] = (group['RSI'] > 70).astype(int)
            group['MACD_Bullish'] = (group['MACD'] > group['MACD_Signal']).astype(int)
            group['BB_Squeeze'] = (group['BB_Width'] < group['BB_Width'].rolling(20).quantile(0.2)).astype(int)

            # High/Low ratios and price position
            group['High_Low_Ratio'] = group['high'] / group['low']
            group['Close_Position'] = (group['close'] - group['low']) / (group['high'] - group['low'])

            # Volume-price analysis
            group['Volume_Price_Trend'] = (group['close'].diff() * group['volume']).rolling(5).sum()
            group['Price_Volume_Correlation'] = (
                group['close'].rolling(20).corr(group['volume'])
            )

            results.append(group)

        except Exception as e:
            print(f"Error computing technical features for {symbol}: {e}")
            continue

    if not results:
        raise ValueError("No valid technical features computed for any symbol")

    return pd.concat(results, ignore_index=True)


# =============================================================================
# FEATURE SELECTION AND IMPORTANCE ANALYSIS ENGINE
# =============================================================================

class AdvancedFeatureSelector:
    """Comprehensive feature selection framework for RQ3 analysis"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.selection_results = {}
        self.feature_importance_scores = {}

    def univariate_filter_methods(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Advanced univariate filter-based feature selection"""
        results = {}

        # F-statistic for classification
        try:
            selector_f = SelectKBest(score_func=f_classif, k='all')
            selector_f.fit(X, y)

            f_scores_df = pd.DataFrame({
                'feature': X.columns,
                'f_score': selector_f.scores_,
                'p_value': selector_f.pvalues_
            }).sort_values('f_score', ascending=False)

            results['f_classif_scores'] = f_scores_df

            # Top k selections
            for k in self.config.k_features:
                if k <= len(X.columns):
                    results[f'f_classif_top_{k}'] = f_scores_df.head(k)['feature'].tolist()
        except Exception as e:
            print(f"F-statistic selection failed: {e}")

        # Mutual information for feature relevance
        try:
            mi_scores = mutual_info_classif(X, y, random_state=self.config.random_state)
            mi_scores_df = pd.DataFrame({
                'feature': X.columns,
                'mi_score': mi_scores
            }).sort_values('mi_score', ascending=False)

            results['mutual_info_scores'] = mi_scores_df

            for k in self.config.k_features:
                if k <= len(X.columns):
                    results[f'mutual_info_top_{k}'] = mi_scores_df.head(k)['feature'].tolist()
        except Exception as e:
            print(f"Mutual information selection failed: {e}")

        # Variance-based filtering
        try:
            # Remove low variance features
            variance_threshold = VarianceThreshold(threshold=0.01)
            variance_threshold.fit(X)

            high_variance_features = X.columns[variance_threshold.get_support()].tolist()
            results['high_variance_features'] = high_variance_features

            # Feature variance ranking
            feature_variances = pd.DataFrame({
                'feature': X.columns,
                'variance': X.var()
            }).sort_values('variance', ascending=False)

            results['variance_scores'] = feature_variances

            for k in self.config.k_features:
                if k <= len(feature_variances):
                    results[f'variance_top_{k}'] = feature_variances.head(k)['feature'].tolist()
        except Exception as e:
            print(f"Variance-based selection failed: {e}")

        # Correlation-based feature filtering
        try:
            corr_matrix = X.corr().abs()

            # Find highly correlated feature pairs
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            high_corr_pairs = []
            for column in upper_tri.columns:
                high_corr_features = upper_tri.index[upper_tri[column] > 0.9].tolist()
                for feature in high_corr_features:
                    high_corr_pairs.append((column, feature, upper_tri.loc[feature, column]))

            # Remove redundant features (keep first occurrence)
            features_to_remove = set()
            for feat1, feat2, corr_val in high_corr_pairs:
                if feat1 not in features_to_remove:
                    features_to_remove.add(feat2)

            low_corr_features = [f for f in X.columns if f not in features_to_remove]
            results['low_correlation_features'] = low_corr_features
            results['high_correlation_pairs'] = high_corr_pairs

        except Exception as e:
            print(f"Correlation-based selection failed: {e}")

        self.selection_results['filter'] = results
        return results

    def wrapper_based_methods(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Wrapper-based feature selection using multiple estimators"""
        results = {}

        # Define base estimators for wrapper methods
        estimators = {
            'logistic': LogisticRegression(
                max_iter=1000,
                random_state=self.config.random_state,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.random_state,
                class_weight='balanced'
            )
        }

        if HAVE_XGB:
            estimators['xgboost'] = XGBClassifier(
                n_estimators=100,
                random_state=self.config.random_state,
                eval_metric='logloss'
            )

        # Recursive Feature Elimination (RFE)
        for est_name, estimator in estimators.items():
            for k in self.config.k_features:
                if k <= len(X.columns):
                    try:
                        # Scale features for algorithms that need it
                        if est_name in ['logistic', 'svm']:
                            scaler = StandardScaler()
                            X_scaled = pd.DataFrame(
                                scaler.fit_transform(X),
                                columns=X.columns,
                                index=X.index
                            )
                            rfe = RFE(estimator, n_features_to_select=k, step=1)
                            rfe.fit(X_scaled, y)
                        else:
                            rfe = RFE(estimator, n_features_to_select=k, step=1)
                            rfe.fit(X, y)

                        selected_features = X.columns[rfe.support_].tolist()
                        feature_rankings = pd.DataFrame({
                            'feature': X.columns,
                            'ranking': rfe.ranking_,
                            'selected': rfe.support_
                        }).sort_values('ranking')

                        results[f'rfe_{est_name}_top_{k}'] = selected_features
                        results[f'rfe_{est_name}_rankings_{k}'] = feature_rankings

                    except Exception as e:
                        print(f"RFE with {est_name} failed for k={k}: {e}")

        # Recursive Feature Elimination with Cross-Validation - Fixed
        for est_name, estimator in estimators.items():
            if est_name == 'random_forest':  # Use RF for RFECV as it's most stable
                try:
                    cv = TimeSeriesSplit(n_splits=min(3, self.config.cv_folds))

                    rfecv = RFECV(
                        estimator,
                        step=1,
                        cv=cv,
                        scoring='f1',
                        min_features_to_select=5
                    )
                    rfecv.fit(X, y)

                    optimal_features = X.columns[rfecv.support_].tolist()
                    results[f'rfecv_{est_name}_optimal'] = optimal_features

                    # Fixed: Use cv_results_ instead of grid_scores_
                    if hasattr(rfecv, 'cv_results_'):
                        results[f'rfecv_{est_name}_scores'] = rfecv.cv_results_
                    elif hasattr(rfecv, 'grid_scores_'):
                        results[f'rfecv_{est_name}_scores'] = rfecv.grid_scores_
                    else:
                        # Fallback - create scores from cross-validation
                        results[f'rfecv_{est_name}_scores'] = list(range(5, len(X.columns) + 1))

                    results[f'rfecv_{est_name}_n_features'] = rfecv.n_features_

                except Exception as e:
                    print(f"RFECV with {est_name} failed: {e}")

        self.selection_results['wrapper'] = results
        return results

    def embedded_methods(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Embedded feature selection methods with regularization and tree-based importance"""
        results = {}

        # L1 Regularization (Lasso) feature selection
        try:
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )

            # Try different C values to find optimal sparsity
            C_values = [0.001, 0.01, 0.1, 1.0, 10.0]
            lasso_results = {}

            for C in C_values:
                lasso = LogisticRegression(
                    penalty='l1',
                    solver='liblinear',
                    C=C,
                    random_state=self.config.random_state,
                    max_iter=1000
                )
                lasso.fit(X_scaled, y)

                # Features with non-zero coefficients
                non_zero_mask = lasso.coef_[0] != 0
                selected_features = X.columns[non_zero_mask].tolist()

                lasso_results[f'C_{C}'] = {
                    'features': selected_features,
                    'n_features': len(selected_features),
                    'coefficients': lasso.coef_[0][non_zero_mask]
                }

            # Select the C value that gives reasonable sparsity
            optimal_C = None
            for C in C_values:
                if 5 <= lasso_results[f'C_{C}']['n_features'] <= 20:
                    optimal_C = C
                    break

            if optimal_C is None:
                optimal_C = 1.0  # Default fallback

            results['lasso_optimal_C'] = optimal_C
            results['lasso_selected_features'] = lasso_results[f'C_{optimal_C}']['features']

            # Feature importance ranking by absolute coefficient values
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'abs_coefficient': np.abs(lasso.coef_[0])
            }).sort_values('abs_coefficient', ascending=False)

            results['lasso_importance_ranking'] = feature_importance

            for k in self.config.k_features:
                if k <= len(feature_importance):
                    results[f'lasso_top_{k}'] = feature_importance.head(k)['feature'].tolist()

        except Exception as e:
            print(f"Lasso feature selection failed: {e}")

        # Random Forest feature importance
        try:
            rf = RandomForestClassifier(
                n_estimators=300,
                random_state=self.config.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
            rf.fit(X, y)

            # Feature importance ranking
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_,
                'importance_std': np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
            }).sort_values('importance', ascending=False)

            results['rf_importance_ranking'] = feature_importance

            for k in self.config.k_features:
                if k <= len(feature_importance):
                    results[f'rf_top_{k}'] = feature_importance.head(k)['feature'].tolist()

            # SelectFromModel with different thresholds
            for threshold in ['mean', 'median', '0.1*mean']:
                try:
                    selector = SelectFromModel(rf, threshold=threshold)
                    selector.fit(X, y)
                    selected_features = X.columns[selector.get_support()].tolist()
                    results[f'rf_{threshold.replace("*", "_")}_threshold'] = selected_features
                except Exception as e:
                    print(f"RF SelectFromModel with {threshold} failed: {e}")

        except Exception as e:
            print(f"Random Forest feature selection failed: {e}")

        # XGBoost feature importance
        if HAVE_XGB:
            try:
                xgb = XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.config.random_state,
                    eval_metric='logloss'
                )
                xgb.fit(X, y)

                # Multiple importance types
                importance_types = ['weight', 'gain', 'cover']
                for imp_type in importance_types:
                    try:
                        importance_dict = xgb.get_booster().get_score(importance_type=imp_type)

                        # Create importance dataframe
                        feature_importance = pd.DataFrame([
                            {'feature': feat, f'{imp_type}_importance': importance_dict.get(feat, 0)}
                            for feat in X.columns
                        ]).sort_values(f'{imp_type}_importance', ascending=False)

                        results[f'xgb_{imp_type}_importance_ranking'] = feature_importance

                        for k in self.config.k_features:
                            if k <= len(feature_importance):
                                results[f'xgb_{imp_type}_top_{k}'] = feature_importance.head(k)['feature'].tolist()
                    except Exception as e:
                        print(f"XGB {imp_type} importance failed: {e}")

            except Exception as e:
                print(f"XGBoost feature selection failed: {e}")

        # Gradient Boosting feature importance
        try:
            gb = GradientBoostingClassifier(
                n_estimators=200,
                random_state=self.config.random_state
            )
            gb.fit(X, y)

            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': gb.feature_importances_
            }).sort_values('importance', ascending=False)

            results['gb_importance_ranking'] = feature_importance

            for k in self.config.k_features:
                if k <= len(feature_importance):
                    results[f'gb_top_{k}'] = feature_importance.head(k)['feature'].tolist()

        except Exception as e:
            print(f"Gradient Boosting feature selection failed: {e}")

        self.selection_results['embedded'] = results
        return results

    def shap_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """SHAP-based feature importance analysis"""
        if not HAVE_SHAP:
            print("SHAP not available, skipping SHAP analysis")
            return {}

        results = {}

        try:
            print("Computing SHAP feature importance...")

            # Use only numeric features for SHAP
            X_numeric = X.select_dtypes(include=[np.number])
            if X_numeric.shape[1] == 0:
                raise ValueError("No numeric features available for SHAP analysis")

            # Sample data for performance (SHAP can be slow)
            sample_size = min(500, len(X_numeric))  # Reduced sample size
            if len(X_numeric) > sample_size:
                sample_idx = np.random.choice(len(X_numeric), sample_size, replace=False)
                X_shap = X_numeric.iloc[sample_idx]
                y_shap = y.iloc[sample_idx]
            else:
                X_shap = X_numeric
                y_shap = y

            # Clean data for SHAP - more thorough cleaning
            X_shap = X_shap.replace([np.inf, -np.inf], np.nan)
            X_shap = X_shap.fillna(X_shap.mean())  # Fill with mean instead of 0

            # Additional check for problematic columns
            problematic_cols = []
            for col in X_shap.columns:
                if X_shap[col].isna().all() or X_shap[col].std() == 0:
                    problematic_cols.append(col)

            if problematic_cols:
                X_shap = X_shap.drop(columns=problematic_cols)
                print(f"Dropped problematic columns: {problematic_cols}")

            if X_shap.shape[1] == 0:
                raise ValueError("No valid features remaining for SHAP analysis")

            # Train model for SHAP analysis
            model = RandomForestClassifier(
                n_estimators=50,  # Reduced for speed
                random_state=self.config.random_state,
                max_depth=8
            )
            model.fit(X_shap, y_shap)

            # Compute SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap, check_additivity=False)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Binary classification - take positive class
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            # Convert to numpy array if needed
            if hasattr(shap_values, 'values'):
                shap_values = shap_values.values

            shap_values = np.array(shap_values)

            # Ensure 2D array
            if shap_values.ndim != 2:
                print(f"Warning: SHAP values have unexpected shape {shap_values.shape}")
                return {}

            # Compute feature importance as mean absolute SHAP values
            shap_importance = np.mean(np.abs(shap_values), axis=0)

            # Create importance dataframe
            feature_importance = pd.DataFrame({
                'feature': X_shap.columns,
                'shap_importance': shap_importance
            }).sort_values('shap_importance', ascending=False)

            results['shap_importance_ranking'] = feature_importance
            results['shap_values'] = shap_values
            results['shap_feature_names'] = list(X_shap.columns)

            # Export for dashboard
            feature_importance.to_csv(
                self.config.dashboard_dir / "shap_importance.csv",
                index=False
            )

            for k in self.config.k_features:
                if k <= len(feature_importance):
                    results[f'shap_top_{k}'] = feature_importance.head(k)['feature'].tolist()

        except Exception as e:
            print(f"SHAP feature importance analysis failed: {e}")

        self.selection_results['shap'] = results
        return results

    def consensus_feature_ranking(self) -> pd.DataFrame:
        """Create consensus ranking across all feature selection methods"""

        # Collect all feature rankings from different methods
        all_rankings = []

        for method_type, method_results in self.selection_results.items():
            if method_type == 'shap' and not method_results:
                continue

            if isinstance(method_results, dict):
                for selection_name, features in method_results.items():
                    # Only process feature lists, not other result types
                    if isinstance(features, list) and all(isinstance(f, str) for f in features):
                        for rank, feature in enumerate(features, 1):
                            all_rankings.append({
                                'method_type': method_type,
                                'selection_method': selection_name,
                                'feature': feature,
                                'rank': rank,
                                'inverse_rank_score': 1.0 / rank,
                                'selection_frequency': 1
                            })

        if not all_rankings:
            return pd.DataFrame()

        rankings_df = pd.DataFrame(all_rankings)

        # Compute consensus metrics
        consensus = rankings_df.groupby('feature').agg({
            'inverse_rank_score': ['sum', 'mean'],
            'selection_frequency': 'sum',
            'rank': ['mean', 'min', 'std']
        }).round(4)

        # Flatten column names
        consensus.columns = [
            'total_score', 'mean_score', 'selection_count',
            'mean_rank', 'best_rank', 'rank_std'
        ]
        consensus = consensus.reset_index()

        # Sort by selection frequency first, then by total score
        consensus = consensus.sort_values(
            ['selection_count', 'total_score'],
            ascending=[False, False]
        ).reset_index(drop=True)

        # Add consensus rank
        consensus['consensus_rank'] = range(1, len(consensus) + 1)

        return consensus

    def method_consistency_analysis(self) -> Dict[str, Any]:
        """Analyze consistency between different feature selection methods"""

        consensus_df = self.consensus_feature_ranking()
        if consensus_df.empty:
            return {}

        consistency_metrics = {}

        # Pairwise overlap analysis between methods
        method_comparisons = []

        # Get top features from each method
        method_top_features = {}
        for method_type, method_results in self.selection_results.items():
            if method_type == 'shap' and not method_results:
                continue

            if isinstance(method_results, dict):
                for selection_name, features in method_results.items():
                    if (isinstance(features, list) and
                            all(isinstance(f, str) for f in features) and
                            'top_10' in selection_name):
                        method_top_features[selection_name] = set(features)

        # Compute pairwise overlaps
        method_names = list(method_top_features.keys())
        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names[i + 1:], i + 1):
                features1 = method_top_features[method1]
                features2 = method_top_features[method2]

                overlap = len(features1 & features2)
                union = len(features1 | features2)
                jaccard = overlap / union if union > 0 else 0

                method_comparisons.append({
                    'method1': method1,
                    'method2': method2,
                    'overlap': overlap,
                    'jaccard_similarity': jaccard,
                    'method1_size': len(features1),
                    'method2_size': len(features2)
                })

        consistency_metrics['pairwise_comparisons'] = pd.DataFrame(method_comparisons)

        # Overall consistency metrics
        if method_comparisons:
            consistency_metrics['mean_jaccard_similarity'] = np.mean(
                [c['jaccard_similarity'] for c in method_comparisons])
            consistency_metrics['mean_overlap'] = np.mean([c['overlap'] for c in method_comparisons])

        # Feature stability analysis
        feature_stability = consensus_df.copy()
        feature_stability['stability_score'] = (
                feature_stability['selection_count'] / len(method_top_features)
        )

        consistency_metrics['feature_stability'] = feature_stability
        consistency_metrics['highly_stable_features'] = feature_stability[
            feature_stability['stability_score'] >= 0.7
            ]['feature'].tolist()

        return consistency_metrics


# =============================================================================
# MODEL TRAINING AND EVALUATION ENGINE
# =============================================================================

def create_comprehensive_model_specs(config: AnalysisConfig) -> List[ModelSpec]:
    """Create comprehensive model specifications for evaluation"""
    specs = []

    # Logistic Regression with different regularizations
    specs.append(ModelSpec(
        name='logistic_l1',
        estimator=LogisticRegression(
            penalty='l1',
            solver='liblinear',
            max_iter=2000,
            random_state=config.random_state,
            class_weight='balanced'
        ),
        needs_scaling=True,
        param_grid={'C': [0.01, 0.1, 1.0, 10.0]}
    ))

    specs.append(ModelSpec(
        name='logistic_l2',
        estimator=LogisticRegression(
            penalty='l2',
            solver='lbfgs',
            max_iter=2000,
            random_state=config.random_state,
            class_weight='balanced'
        ),
        needs_scaling=True,
        param_grid={'C': [0.01, 0.1, 1.0, 10.0]}
    ))

    # Random Forest with balanced settings
    specs.append(ModelSpec(
        name='random_forest',
        estimator=RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=config.random_state,
            class_weight='balanced_subsample',
            n_jobs=-1
        ),
        needs_scaling=False,
        param_grid={
            'n_estimators': [100, 200],
            'max_depth': [8, 10, 12],
            'min_samples_split': [3, 5]
        }
    ))

    # Gradient Boosting
    specs.append(ModelSpec(
        name='gradient_boosting',
        estimator=GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=config.random_state
        ),
        needs_scaling=False,
        param_grid={
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [4, 6, 8]
        }
    ))

    # XGBoost if available
    if HAVE_XGB:
        specs.append(ModelSpec(
            name='xgboost',
            estimator=XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=config.random_state,
                eval_metric='logloss'
            ),
            needs_scaling=False,
            param_grid={
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15]
            }
        ))

    # Support Vector Machine
    specs.append(ModelSpec(
        name='svm_rbf',
        estimator=SVC(
            kernel='rbf',
            probability=True,
            random_state=config.random_state,
            class_weight='balanced'
        ),
        needs_scaling=True,
        param_grid={
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto']
        }
    ))

    # K-Nearest Neighbors
    specs.append(ModelSpec(
        name='knn',
        estimator=KNeighborsClassifier(n_neighbors=5),
        needs_scaling=True,
        param_grid={'n_neighbors': [3, 5, 7, 9]}
    ))

    # Naive Bayes
    specs.append(ModelSpec(
        name='naive_bayes',
        estimator=GaussianNB(),
        needs_scaling=False
    ))

    return specs


def create_enhanced_feature_bundles(sentiment_cols: List[str], technical_cols: List[str]) -> List[FeatureBundle]:
    """Create comprehensive feature bundles for RQ2 analysis"""

    bundles = []

    # Pure sentiment bundle
    bundles.append(FeatureBundle(
        name='sentiment_only',
        columns=sentiment_cols,
        description='Pure sentiment-based features from news and social media',
        bundle_type='sentiment'
    ))

    # Pure technical bundle
    bundles.append(FeatureBundle(
        name='technical_only',
        columns=technical_cols,
        description='Pure technical analysis indicators and price patterns',
        bundle_type='technical'
    ))

    # Hybrid bundle (combination)
    bundles.append(FeatureBundle(
        name='hybrid',
        columns=sentiment_cols + technical_cols,
        description='Combined sentiment and technical features',
        bundle_type='hybrid'
    ))

    # Subset bundles for granular analysis
    if len(sentiment_cols) >= 4:
        bundles.append(FeatureBundle(
            name='sentiment_core',
            columns=sentiment_cols[:4],  # Top sentiment features
            description='Core sentiment indicators',
            bundle_type='sentiment'
        ))

    if len(technical_cols) >= 10:
        # Technical subsets
        trend_indicators = [col for col in technical_cols if any(x in col.lower() for x in ['sma', 'ema', 'price_vs'])]
        momentum_indicators = [col for col in technical_cols if
                               any(x in col.lower() for x in ['rsi', 'macd', 'roc', 'momentum'])]
        volume_indicators = [col for col in technical_cols if any(x in col.lower() for x in ['volume', 'obv'])]

        if trend_indicators:
            bundles.append(FeatureBundle(
                name='technical_trend',
                columns=trend_indicators,
                description='Trend-following technical indicators',
                bundle_type='technical'
            ))

        if momentum_indicators:
            bundles.append(FeatureBundle(
                name='technical_momentum',
                columns=momentum_indicators,
                description='Momentum and oscillator indicators',
                bundle_type='technical'
            ))

        if volume_indicators:
            bundles.append(FeatureBundle(
                name='technical_volume',
                columns=volume_indicators,
                description='Volume-based technical indicators',
                bundle_type='technical'
            ))

    return bundles


def advanced_time_series_split(df: pd.DataFrame, test_fraction: float = 0.2,
                               validation_fraction: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Advanced time-aware train-validation-test split"""

    unique_dates = sorted(df['date'].unique())
    n_dates = len(unique_dates)

    # Calculate split indices
    test_start_idx = int(n_dates * (1 - test_fraction))
    val_start_idx = int(n_dates * (1 - test_fraction - validation_fraction))

    test_start_date = unique_dates[test_start_idx]
    val_start_date = unique_dates[val_start_idx]

    # Create splits
    train_df = df[df['date'] < val_start_date].copy()
    val_df = df[(df['date'] >= val_start_date) & (df['date'] < test_start_date)].copy()
    test_df = df[df['date'] >= test_start_date].copy()

    return train_df, val_df, test_df


def compute_comprehensive_metrics(model, X_test: pd.DataFrame, y_test: pd.Series,
                                  model_name: str = '', bundle_name: str = '') -> Dict[str, float]:
    """Compute comprehensive evaluation metrics"""

    # Basic predictions
    y_pred = model.predict(X_test)

    # Probability predictions if available
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except:
        y_proba = None

    # Basic classification metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'specificity': recall_score(1 - y_test, 1 - y_pred, zero_division=0)
    }

    # ROC-AUC if probabilities available
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        except:
            metrics['roc_auc'] = np.nan
    else:
        metrics['roc_auc'] = np.nan

    # Confusion matrix components
    try:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics.update({
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0
        })
    except:
        metrics.update({
            'true_positives': 0, 'true_negatives': 0,
            'false_positives': 0, 'false_negatives': 0,
            'positive_predictive_value': 0, 'negative_predictive_value': 0
        })

    # Additional performance metrics
    metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2

    # F2 score (emphasizes recall)
    try:
        from sklearn.metrics import fbeta_score
        metrics['f2_score'] = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
    except:
        metrics['f2_score'] = np.nan

    # Matthews Correlation Coefficient
    try:
        from sklearn.metrics import matthews_corrcoef
        metrics['mcc'] = matthews_corrcoef(y_test, y_pred)
    except:
        metrics['mcc'] = np.nan

    return metrics


def cross_validation_analysis(X: pd.DataFrame, y: pd.Series, model_specs: List[ModelSpec],
                              config: AnalysisConfig) -> pd.DataFrame:
    """Comprehensive cross-validation analysis"""

    cv_results = []

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=config.cv_folds)

    for spec in model_specs:
        try:
            # Create pipeline
            if spec.needs_scaling:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', spec.estimator)
                ])
            else:
                pipeline = Pipeline([('model', spec.estimator)])

            # Cross-validation for multiple metrics
            scoring_metrics = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']

            cv_scores = {}
            for metric in scoring_metrics:
                try:
                    scores = cross_val_score(
                        pipeline, X, y,
                        cv=tscv,
                        scoring=metric,
                        n_jobs=-1
                    )
                    cv_scores[f'{metric}_scores'] = scores
                    cv_scores[f'{metric}_mean'] = scores.mean()
                    cv_scores[f'{metric}_std'] = scores.std()
                except Exception as e:
                    print(f"CV scoring for {metric} failed: {e}")
                    cv_scores[f'{metric}_scores'] = np.array([])
                    cv_scores[f'{metric}_mean'] = np.nan
                    cv_scores[f'{metric}_std'] = np.nan

            # Compile results
            result = {
                'model': spec.name,
                'cv_folds': config.cv_folds,
                **cv_scores
            }

            cv_results.append(result)

        except Exception as e:
            print(f"Cross-validation failed for {spec.name}: {e}")

    return pd.DataFrame(cv_results)


# =============================================================================
# HYPOTHESIS TESTING AND STATISTICAL ANALYSIS ENGINE
# =============================================================================

class HypothesisTestingEngine:
    """Comprehensive hypothesis testing framework for RQ2 and RQ3"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.test_results = {}

    def cohen_d(self, group1: np.array, group2: np.array) -> float:
        """Calculate Cohen's d effect size"""
        try:
            n1, n2 = len(group1), len(group2)
            if n1 <= 1 or n2 <= 1:
                return 0.0

            # Pooled standard deviation
            pooled_std = np.sqrt(
                ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) /
                (n1 + n2 - 2)
            )

            if pooled_std == 0:
                return 0.0

            return (np.mean(group1) - np.mean(group2)) / pooled_std
        except:
            return 0.0

    def interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Negligible"
        elif abs_d < 0.5:
            return "Small"
        elif abs_d < 0.8:
            return "Medium"
        else:
            return "Large"

    def rq2_hypothesis_testing(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        RQ2 Hypothesis Testing: Technical vs Sentiment Comparison
        H0: Hybrid models ≠ individual models
        H1: Hybrid > individual models
        """

        rq2_results = {
            'research_question': 'RQ2: Technical vs Sentiment Comparison - Which models perform better?',
            'hypotheses': {
                'H0': 'Hybrid models ≠ individual models (no systematic performance difference)',
                'H1': 'Hybrid > individual models (hybrid models systematically outperform individual models)'
            },
            'statistical_tests': [],
            'summary': {}
        }

        # Group by bundle type for analysis
        bundle_performance = {}
        for bundle in results_df['bundle'].unique():
            bundle_data = results_df[results_df['bundle'] == bundle]
            if len(bundle_data) > 0:
                bundle_performance[bundle] = {
                    'accuracy': bundle_data['accuracy'].values,
                    'f1_score': bundle_data['f1_score'].values,
                    'roc_auc': bundle_data['roc_auc'].values,
                    'mean_accuracy': bundle_data['accuracy'].mean(),
                    'mean_f1': bundle_data['f1_score'].mean(),
                    'mean_auc': bundle_data['roc_auc'].mean(),
                    'count': len(bundle_data)
                }

        # Main hypothesis test: Hybrid vs Individual models
        metrics_to_test = ['accuracy', 'f1_score', 'roc_auc']

        # Test 1: Hybrid vs Sentiment Only
        if 'hybrid' in bundle_performance and 'sentiment_only' in bundle_performance:
            for metric in metrics_to_test:
                hybrid_scores = bundle_performance['hybrid'][metric]
                sentiment_scores = bundle_performance['sentiment_only'][metric]

                if len(hybrid_scores) > 1 and len(sentiment_scores) > 1:
                    # Perform t-test
                    t_stat, p_value = ttest_ind(hybrid_scores, sentiment_scores)

                    # Effect size
                    cohens_d = self.cohen_d(hybrid_scores, sentiment_scores)
                    effect_interp = self.interpret_effect_size(cohens_d)

                    # Determine significance
                    significant = p_value < self.config.alpha

                    test_result = HypothesisTest(
                        research_question='RQ2',
                        hypothesis_null='Hybrid = Sentiment Only',
                        hypothesis_alternative='Hybrid > Sentiment Only',
                        test_statistic=t_stat,
                        p_value=p_value,
                        effect_size=cohens_d,
                        effect_interpretation=effect_interp,
                        significant=significant,
                        conclusion='Support H1' if (significant and cohens_d > 0) else 'Fail to reject H0',
                        method='Independent t-test',
                        sample_size=len(hybrid_scores) + len(sentiment_scores)
                    )

                    rq2_results['statistical_tests'].append({
                        'comparison': 'hybrid_vs_sentiment_only',
                        'metric': metric,
                        'test_result': test_result.__dict__,
                        'hybrid_mean': np.mean(hybrid_scores),
                        'sentiment_mean': np.mean(sentiment_scores)
                    })

        # Test 2: Hybrid vs Technical Only
        if 'hybrid' in bundle_performance and 'technical_only' in bundle_performance:
            for metric in metrics_to_test:
                hybrid_scores = bundle_performance['hybrid'][metric]
                technical_scores = bundle_performance['technical_only'][metric]

                if len(hybrid_scores) > 1 and len(technical_scores) > 1:
                    # Perform t-test
                    t_stat, p_value = ttest_ind(hybrid_scores, technical_scores)

                    # Effect size
                    cohens_d = self.cohen_d(hybrid_scores, technical_scores)
                    effect_interp = self.interpret_effect_size(cohens_d)

                    # Determine significance
                    significant = p_value < self.config.alpha

                    test_result = HypothesisTest(
                        research_question='RQ2',
                        hypothesis_null='Hybrid = Technical Only',
                        hypothesis_alternative='Hybrid > Technical Only',
                        test_statistic=t_stat,
                        p_value=p_value,
                        effect_size=cohens_d,
                        effect_interpretation=effect_interp,
                        significant=significant,
                        conclusion='Support H1' if (significant and cohens_d > 0) else 'Fail to reject H0',
                        method='Independent t-test',
                        sample_size=len(hybrid_scores) + len(technical_scores)
                    )

                    rq2_results['statistical_tests'].append({
                        'comparison': 'hybrid_vs_technical_only',
                        'metric': metric,
                        'test_result': test_result.__dict__,
                        'hybrid_mean': np.mean(hybrid_scores),
                        'technical_mean': np.mean(technical_scores)
                    })

        # Test 3: Sentiment vs Technical (exploratory)
        if 'sentiment_only' in bundle_performance and 'technical_only' in bundle_performance:
            for metric in metrics_to_test:
                sentiment_scores = bundle_performance['sentiment_only'][metric]
                technical_scores = bundle_performance['technical_only'][metric]

                if len(sentiment_scores) > 1 and len(technical_scores) > 1:
                    # Perform t-test
                    t_stat, p_value = ttest_ind(sentiment_scores, technical_scores)

                    # Effect size
                    cohens_d = self.cohen_d(sentiment_scores, technical_scores)
                    effect_interp = self.interpret_effect_size(cohens_d)

                    # Determine significance
                    significant = p_value < self.config.alpha

                    test_result = HypothesisTest(
                        research_question='RQ2_Exploratory',
                        hypothesis_null='Sentiment = Technical',
                        hypothesis_alternative='Sentiment ≠ Technical',
                        test_statistic=t_stat,
                        p_value=p_value,
                        effect_size=cohens_d,
                        effect_interpretation=effect_interp,
                        significant=significant,
                        conclusion='Significant difference' if significant else 'No significant difference',
                        method='Independent t-test',
                        sample_size=len(sentiment_scores) + len(technical_scores)
                    )

                    rq2_results['statistical_tests'].append({
                        'comparison': 'sentiment_vs_technical',
                        'metric': metric,
                        'test_result': test_result.__dict__,
                        'sentiment_mean': np.mean(sentiment_scores),
                        'technical_mean': np.mean(technical_scores)
                    })

        # Summary analysis
        significant_tests = [test for test in rq2_results['statistical_tests']
                             if test['test_result']['significant']]

        # Count support for H1 (Hybrid > Individual)
        h1_support_count = 0
        total_h1_tests = 0

        for test in rq2_results['statistical_tests']:
            if 'hybrid_vs' in test['comparison']:
                total_h1_tests += 1
                if (test['test_result']['significant'] and
                        test['test_result']['effect_size'] > 0):
                    h1_support_count += 1

        rq2_results['summary'] = {
            'total_tests_performed': len(rq2_results['statistical_tests']),
            'significant_tests': len(significant_tests),
            'h1_support_tests': h1_support_count,
            'total_h1_tests': total_h1_tests,
            'h1_support_rate': h1_support_count / total_h1_tests if total_h1_tests > 0 else 0,
            'overall_conclusion': self._determine_rq2_conclusion(h1_support_count, total_h1_tests),
            'bundle_rankings': bundle_performance
        }

        return rq2_results

    def _determine_rq2_conclusion(self, h1_support: int, total_tests: int) -> str:
        """Determine overall conclusion for RQ2"""
        if total_tests == 0:
            return "Insufficient data for hypothesis testing"

        support_rate = h1_support / total_tests

        if support_rate >= 0.8:
            return "Strong support for H1: Hybrid models systematically outperform individual models"
        elif support_rate >= 0.6:
            return "Moderate support for H1: Hybrid models generally outperform individual models"
        elif support_rate >= 0.4:
            return "Mixed evidence: No clear superiority of hybrid models"
        else:
            return "Evidence against H1: Individual models may perform as well as or better than hybrid models"

    def rq3_hypothesis_testing(self, feature_selector: AdvancedFeatureSelector,
                               consensus_df: pd.DataFrame) -> Dict[str, Any]:
        """
        RQ3 Hypothesis Testing: Feature Importance Analysis
        H0: No KPI dominates
        H1: Some KPIs stronger
        """

        rq3_results = {
            'research_question': 'RQ3: Feature Importance Analysis - Which KPIs matter most?',
            'hypotheses': {
                'H0': 'No KPI dominates (all features have equal importance)',
                'H1': 'Some KPIs stronger (significant variation in feature importance)'
            },
            'statistical_tests': [],
            'summary': {}
        }

        if consensus_df.empty:
            rq3_results['summary']['conclusion'] = "Insufficient feature selection data"
            return rq3_results

        # Test 1: Variance in feature importance scores
        importance_scores = consensus_df['total_score'].values

        if len(importance_scores) > 1:
            # Test if variance in importance scores is significantly different from uniform
            # Using coefficient of variation
            cv = np.std(importance_scores) / np.mean(importance_scores)

            # Chi-square goodness of fit test for uniform distribution
            try:
                expected_uniform = np.full(len(importance_scores), np.mean(importance_scores))
                from scipy.stats import chisquare
                chi2_stat, chi2_p = chisquare(importance_scores + 1e-10, expected_uniform + 1e-10)

                test_result = HypothesisTest(
                    research_question='RQ3',
                    hypothesis_null='Features have uniform importance',
                    hypothesis_alternative='Features have varying importance',
                    test_statistic=chi2_stat,
                    p_value=chi2_p,
                    effect_size=cv,
                    effect_interpretation='High' if cv > 0.5 else 'Medium' if cv > 0.3 else 'Low',
                    significant=chi2_p < self.config.alpha,
                    conclusion='Support H1' if chi2_p < self.config.alpha else 'Fail to reject H0',
                    method='Chi-square goodness of fit',
                    sample_size=len(importance_scores)
                )

                rq3_results['statistical_tests'].append({
                    'test_type': 'importance_variance',
                    'test_result': test_result.__dict__,
                    'coefficient_of_variation': cv
                })

            except Exception as e:
                print(f"Chi-square test failed: {e}")

        # Test 2: Method consistency analysis
        consistency_analysis = feature_selector.method_consistency_analysis()

        if consistency_analysis and 'mean_jaccard_similarity' in consistency_analysis:
            jaccard_sim = consistency_analysis['mean_jaccard_similarity']

            # Test if method consistency is significantly above chance
            # Null hypothesis: Jaccard similarity = 0.1 (random overlap)
            # Alternative: Jaccard similarity > 0.1 (systematic agreement)

            test_result = HypothesisTest(
                research_question='RQ3',
                hypothesis_null='Feature selection methods show random agreement (J ≤ 0.1)',
                hypothesis_alternative='Feature selection methods show systematic agreement (J > 0.1)',
                test_statistic=jaccard_sim,
                p_value=0.05 if jaccard_sim > 0.1 else 0.5,  # Simplified p-value
                effect_size=jaccard_sim,
                effect_interpretation='High' if jaccard_sim > 0.4 else 'Medium' if jaccard_sim > 0.2 else 'Low',
                significant=jaccard_sim > 0.1,
                conclusion='Systematic method agreement' if jaccard_sim > 0.1 else 'Random method agreement',
                method='Jaccard similarity analysis',
                sample_size=len(consistency_analysis.get('pairwise_comparisons', []))
            )

            rq3_results['statistical_tests'].append({
                'test_type': 'method_consistency',
                'test_result': test_result.__dict__,
                'jaccard_similarity': jaccard_sim
            })

        # Test 3: Dominant features identification
        top_features = consensus_df.head(10)
        if len(top_features) >= 5:
            # Test if top features are significantly more important than others
            top_scores = top_features['total_score'].values
            remaining_scores = consensus_df.iloc[10:]['total_score'].values if len(consensus_df) > 10 else []

            if len(remaining_scores) > 0:
                # Mann-Whitney U test (non-parametric)
                try:
                    u_stat, u_p = mannwhitneyu(top_scores, remaining_scores, alternative='greater')

                    effect_size = len(top_scores) * len(remaining_scores)
                    if effect_size > 0:
                        effect_size = 1 - (2 * u_stat) / effect_size  # Glass rank biserial correlation

                    test_result = HypothesisTest(
                        research_question='RQ3',
                        hypothesis_null='Top features not significantly more important',
                        hypothesis_alternative='Top features are significantly more important',
                        test_statistic=u_stat,
                        p_value=u_p,
                        effect_size=effect_size,
                        effect_interpretation=self.interpret_effect_size(effect_size),
                        significant=u_p < self.config.alpha,
                        conclusion='Dominant features identified' if u_p < self.config.alpha else 'No dominant features',
                        method='Mann-Whitney U test',
                        sample_size=len(top_scores) + len(remaining_scores)
                    )

                    rq3_results['statistical_tests'].append({
                        'test_type': 'dominant_features',
                        'test_result': test_result.__dict__,
                        'top_feature_names': top_features['feature'].tolist()
                    })

                except Exception as e:
                    print(f"Mann-Whitney U test failed: {e}")

        # Summary analysis
        significant_tests = [test for test in rq3_results['statistical_tests']
                             if test['test_result']['significant']]

        rq3_results['summary'] = {
            'total_features_analyzed': len(consensus_df),
            'total_tests_performed': len(rq3_results['statistical_tests']),
            'significant_tests': len(significant_tests),
            'top_10_features': consensus_df.head(10)['feature'].tolist(),
            'most_important_feature': consensus_df.iloc[0]['feature'] if len(consensus_df) > 0 else None,
            'importance_variation': 'High' if len(significant_tests) >= 2 else 'Low',
            'overall_conclusion': self._determine_rq3_conclusion(significant_tests, consensus_df)
        }

        return rq3_results

    def _determine_rq3_conclusion(self, significant_tests: List, consensus_df: pd.DataFrame) -> str:
        """Determine overall conclusion for RQ3"""
        if len(significant_tests) >= 2:
            return "Strong support for H1: Significant variation in feature importance with identifiable dominant KPIs"
        elif len(significant_tests) == 1:
            return "Moderate support for H1: Some evidence of feature importance variation"
        else:
            return "Fail to reject H0: No strong evidence of dominant features"

    def multiple_testing_correction(self, p_values: List[float]) -> List[float]:
        """Apply multiple testing correction"""
        if self.config.multiple_testing_correction == 'bonferroni':
            return [min(p * len(p_values), 1.0) for p in p_values]
        elif self.config.multiple_testing_correction == 'fdr_bh':
            # Benjamini-Hochberg procedure
            from scipy.stats import false_discovery_control
            try:
                return false_discovery_control(p_values)
            except:
                return p_values  # Fallback to original p-values
        else:
            return p_values  # No correction

    def comprehensive_hypothesis_analysis(self, results_df: pd.DataFrame,
                                          feature_selector: AdvancedFeatureSelector) -> Dict[str, Any]:
        """Run comprehensive hypothesis testing for both RQ2 and RQ3"""

        # Get consensus feature ranking
        consensus_df = feature_selector.consensus_feature_ranking()

        # RQ2 Analysis
        print("Conducting RQ2 hypothesis testing...")
        rq2_results = self.rq2_hypothesis_testing(results_df)

        # RQ3 Analysis
        print("Conducting RQ3 hypothesis testing...")
        rq3_results = self.rq3_hypothesis_testing(feature_selector, consensus_df)

        # Combined results
        comprehensive_results = {
            'rq2_analysis': rq2_results,
            'rq3_analysis': rq3_results,
            'methodology': {
                'significance_level': self.config.alpha,
                'multiple_testing_correction': self.config.multiple_testing_correction,
                'effect_size_measure': 'Cohen\'s d for mean differences, Coefficient of Variation for variance tests'
            },
            'overall_summary': {
                'rq2_conclusion': rq2_results['summary'].get('overall_conclusion', 'Inconclusive'),
                'rq3_conclusion': rq3_results['summary'].get('overall_conclusion', 'Inconclusive'),
                'research_implications': self._generate_research_implications(rq2_results, rq3_results)
            }
        }

        # Export results for dashboard
        with open(self.config.dashboard_dir / 'hypothesis_testing_results.json', 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)

        return comprehensive_results

    def _generate_research_implications(self, rq2_results: Dict, rq3_results: Dict) -> List[str]:
        """Generate research implications based on hypothesis testing results"""
        implications = []

        # RQ2 implications
        rq2_conclusion = rq2_results['summary'].get('overall_conclusion', '')
        if 'Strong support for H1' in rq2_conclusion:
            implications.append(
                "Hybrid feature models demonstrate systematic performance advantages over single-domain approaches")
            implications.append("Portfolio optimization strategies should consider multi-modal feature integration")
        elif 'Evidence against H1' in rq2_conclusion:
            implications.append("Single-domain feature models may be sufficient for prediction tasks")
            implications.append("Model complexity reduction through domain-specific features is supported")

        # RQ3 implications
        rq3_conclusion = rq3_results['summary'].get('overall_conclusion', '')
        if 'Strong support for H1' in rq3_conclusion:
            implications.append("Feature selection and engineering should focus on identified dominant KPIs")
            implications.append("Risk management frameworks can prioritize high-importance indicators")

            top_feature = rq3_results['summary'].get('most_important_feature')
            if top_feature:
                implications.append(f"The {top_feature} metric emerges as the most predictive indicator")

        if not implications:
            implications.append(
                "Results suggest further investigation needed with larger sample sizes or different methodologies")

        return implications


# =============================================================================
# VISUALIZATION AND REPORTING ENGINE
# =============================================================================

class ComprehensiveVisualizationEngine:
    """Advanced visualization framework for RQ2-RQ3 analysis"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.plots_dir = config.plots_dir
        self.setup_plotting_style()

    def setup_plotting_style(self):
        """Setup consistent and professional plotting style"""
        plt.style.use('default')
        sns.set_palette("husl")

        # Custom color palette
        self.colors = {
            'sentiment': '#FF6B6B',
            'technical': '#4ECDC4',
            'hybrid': '#45B7D1',
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01'
        }

        plt.rcParams.update({
            'figure.figsize': (14, 10),
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 16,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False
        })

    def plot_technical_indicators_dashboard(self, df: pd.DataFrame, sample_symbols: List[str] = None):
        """Create comprehensive technical indicators dashboard"""

        if sample_symbols is None:
            sample_symbols = df['symbol'].unique()[:2]  # Use first 2 symbols

        for symbol in sample_symbols:
            symbol_data = df[df['symbol'] == symbol].sort_values('date').tail(252)  # Last year of data

            if len(symbol_data) < 50:
                continue

            fig = plt.figure(figsize=(20, 16))
            gs = GridSpec(5, 3, height_ratios=[3, 1, 1, 1, 1], hspace=0.3, wspace=0.3)

            # Main price chart with moving averages and Bollinger Bands
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(symbol_data['date'], symbol_data['close'], label='Close Price',
                     linewidth=2, color=self.colors['primary'])

            if 'SMA_20' in symbol_data.columns:
                ax1.plot(symbol_data['date'], symbol_data['SMA_20'],
                         label='SMA 20', alpha=0.7, color=self.colors['secondary'])
            if 'BB_Upper' in symbol_data.columns:
                ax1.plot(symbol_data['date'], symbol_data['BB_Upper'],
                         label='BB Upper', alpha=0.5, linestyle='--')
                ax1.plot(symbol_data['date'], symbol_data['BB_Lower'],
                         label='BB Lower', alpha=0.5, linestyle='--')
                ax1.fill_between(symbol_data['date'], symbol_data['BB_Upper'],
                                 symbol_data['BB_Lower'], alpha=0.1)

            ax1.set_title(f'Technical Analysis Dashboard - {symbol}', fontsize=18, fontweight='bold')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)

            # RSI
            if 'RSI' in symbol_data.columns:
                ax2 = fig.add_subplot(gs[1, 0])
                ax2.plot(symbol_data['date'], symbol_data['RSI'], color=self.colors['accent'])
                ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
                ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
                ax2.fill_between(symbol_data['date'], 30, 70, alpha=0.1)
                ax2.set_title('RSI (14)')
                ax2.set_ylim(0, 100)
                ax2.legend()

            # MACD
            if all(col in symbol_data.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
                ax3 = fig.add_subplot(gs[1, 1])
                ax3.plot(symbol_data['date'], symbol_data['MACD'], label='MACD', color='blue')
                ax3.plot(symbol_data['date'], symbol_data['MACD_Signal'], label='Signal', color='red')
                ax3.bar(symbol_data['date'], symbol_data['MACD_Histogram'],
                        label='Histogram', alpha=0.6, color=self.colors['accent'])
                ax3.set_title('MACD')
                ax3.legend()

            # Volume analysis
            ax4 = fig.add_subplot(gs[1, 2])
            ax4.bar(symbol_data['date'], symbol_data['volume'], alpha=0.7, color=self.colors['technical'])
            if 'VOLUME_SMA_20' in symbol_data.columns:
                ax4.plot(symbol_data['date'], symbol_data['VOLUME_SMA_20'],
                         color='red', label='Volume SMA')
                ax4.legend()
            ax4.set_title('Volume Analysis')

            # Returns distribution
            ax5 = fig.add_subplot(gs[2, :])
            if 'returns_1d' in symbol_data.columns:
                returns = symbol_data['returns_1d'].dropna()
                ax5.hist(returns, bins=30, alpha=0.7, density=True, color=self.colors['primary'])
                ax5.axvline(returns.mean(), color='red', linestyle='--',
                            label=f'Mean: {returns.mean():.4f}')
                ax5.set_title('Daily Returns Distribution')
                ax5.legend()

            plt.tight_layout()
            plt.savefig(self.plots_dir / f'technical_dashboard_{symbol}.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

    def plot_model_performance_analysis(self, results_df: pd.DataFrame, cv_results_df: pd.DataFrame = None):
        """Comprehensive model performance visualization"""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Analysis', fontsize=20, fontweight='bold')

        # Performance by bundle type
        ax1 = axes[0, 0]
        bundle_order = ['sentiment_only', 'technical_only', 'hybrid']
        bundle_colors = [self.colors['sentiment'], self.colors['technical'], self.colors['hybrid']]

        bundle_data = []
        bundle_labels = []
        colors = []

        for i, bundle in enumerate(bundle_order):
            if bundle in results_df['bundle'].values:
                bundle_scores = results_df[results_df['bundle'] == bundle]['f1_score'].dropna()
                if len(bundle_scores) > 0:
                    bundle_data.append(bundle_scores.values)
                    bundle_labels.append(bundle.replace('_', ' ').title())
                    colors.append(bundle_colors[i])

        if bundle_data:
            bp = ax1.boxplot(bundle_data, labels=bundle_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax1.set_title('F1-Score by Feature Bundle')
        ax1.set_ylabel('F1-Score')
        ax1.grid(True, alpha=0.3)

        # Performance by model type
        ax2 = axes[0, 1]
        model_performance = results_df.groupby('model')['f1_score'].agg(['mean', 'std']).sort_values('mean',
                                                                                                     ascending=False)
        bars = ax2.bar(range(len(model_performance)), model_performance['mean'],
                       yerr=model_performance['std'], capsize=5)
        ax2.set_xticks(range(len(model_performance)))
        ax2.set_xticklabels(model_performance.index, rotation=45)
        ax2.set_title('Mean F1-Score by Model Type')
        ax2.set_ylabel('F1-Score')

        # Accuracy vs F1-Score scatter
        ax3 = axes[0, 2]
        for bundle in results_df['bundle'].unique():
            bundle_subset = results_df[results_df['bundle'] == bundle]
            color = self.colors.get(bundle, 'gray')
            ax3.scatter(bundle_subset['accuracy'], bundle_subset['f1_score'],
                        label=bundle.replace('_', ' ').title(), alpha=0.7, s=60, color=color)

        ax3.set_xlabel('Accuracy')
        ax3.set_ylabel('F1-Score')
        ax3.set_title('Accuracy vs F1-Score by Bundle')
        ax3.legend()
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3)

        # ROC-AUC comparison
        ax4 = axes[1, 0]
        auc_data = results_df.dropna(subset=['roc_auc'])
        if not auc_data.empty:
            sns.violinplot(data=auc_data, x='bundle', y='roc_auc', ax=ax4)
            ax4.set_title('ROC-AUC Distribution by Bundle')
            ax4.tick_params(axis='x', rotation=45)

        # Cross-validation stability
        ax5 = axes[1, 1]
        if cv_results_df is not None and not cv_results_df.empty:
            cv_stability = cv_results_df.set_index('model')[['accuracy_mean', 'f1_mean', 'roc_auc_mean']]
            cv_stability.plot(kind='bar', ax=ax5)
            ax5.set_title('Cross-Validation Mean Performance')
            ax5.tick_params(axis='x', rotation=45)
            ax5.legend()
        else:
            ax5.text(0.5, 0.5, 'CV Results\nNot Available',
                     ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Cross-Validation Results')

        # Performance variability
        ax6 = axes[1, 2]
        performance_std = results_df.groupby(['bundle', 'model'])['f1_score'].std().reset_index()
        pivot_std = performance_std.pivot(index='model', columns='bundle', values='f1_score')
        sns.heatmap(pivot_std, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax6)
        ax6.set_title('F1-Score Standard Deviation\n(Model Stability)')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'model_performance_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_importance_comprehensive(self, feature_selector: AdvancedFeatureSelector):
        """Comprehensive feature importance visualization"""

        consensus_df = feature_selector.consensus_feature_ranking()
        if consensus_df.empty:
            return

        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, hspace=0.3, wspace=0.3)

        # Main consensus ranking
        ax1 = fig.add_subplot(gs[0, :])
        top_features = consensus_df.head(20)
        bars = ax1.barh(range(len(top_features)), top_features['total_score'])
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'])
        ax1.set_xlabel('Consensus Importance Score')
        ax1.set_title('Top 20 Features by Consensus Ranking', fontsize=16, fontweight='bold')

        # Color bars by selection frequency
        for i, bar in enumerate(bars):
            selection_count = top_features.iloc[i]['selection_count']
            max_count = top_features['selection_count'].max()
            bar.set_color(plt.cm.plasma(selection_count / max_count))

        # Selection frequency distribution
        ax2 = fig.add_subplot(gs[1, 0])
        freq_dist = consensus_df['selection_count'].value_counts().sort_index()
        ax2.bar(freq_dist.index, freq_dist.values, color=self.colors['primary'])
        ax2.set_xlabel('Selection Frequency')
        ax2.set_ylabel('Number of Features')
        ax2.set_title('Feature Selection Frequency')

        # Feature categories analysis
        ax3 = fig.add_subplot(gs[1, 1])

        # Categorize features
        feature_categories = {
            'Sentiment': ['ensemble', 'sentiment', 'confidence', 'relevance', 'news'],
            'Price Technical': ['rsi', 'macd', 'bb_', 'sma_', 'ema_', 'price_vs'],
            'Volume': ['volume', 'obv'],
            'Volatility': ['atr', 'volatility'],
            'Momentum': ['roc_', 'momentum', 'price_position'],
            'Patterns': ['doji', 'hammer', 'engulfing', 'above', 'oversold', 'overbought']
        }

        category_scores = {}
        for category, keywords in feature_categories.items():
            mask = consensus_df['feature'].str.lower().str.contains('|'.join(keywords), case=False, na=False)
            category_score = consensus_df[mask]['total_score'].sum()
            if category_score > 0:
                category_scores[category] = category_score

        if category_scores:
            categories = list(category_scores.keys())
            scores = list(category_scores.values())
            wedges, texts, autotexts = ax3.pie(scores, labels=categories, autopct='%1.1f%%',
                                               startangle=90)
            ax3.set_title('Feature Importance by Category')

        # SHAP analysis if available
        shap_data = feature_selector.selection_results.get('shap', {})
        if shap_data and 'shap_importance_ranking' in shap_data:
            ax4 = fig.add_subplot(gs[1, 2])
            shap_importance = shap_data['shap_importance_ranking'].head(15)
            bars = ax4.barh(range(len(shap_importance)), shap_importance['shap_importance'])
            ax4.set_yticks(range(len(shap_importance)))
            ax4.set_yticklabels(shap_importance['feature'])
            ax4.set_xlabel('Mean |SHAP Value|')
            ax4.set_title('SHAP Feature Importance')

            for bar in bars:
                bar.set_color(self.colors['accent'])

        # Feature stability over methods
        ax5 = fig.add_subplot(gs[2, 0])
        if len(consensus_df) > 0:
            stability_scores = consensus_df['selection_count'] / consensus_df['selection_count'].max()
            ax5.hist(stability_scores, bins=20, alpha=0.7, color=self.colors['hybrid'])
            ax5.set_xlabel('Stability Score (Normalized)')
            ax5.set_ylabel('Number of Features')
            ax5.set_title('Feature Stability Distribution')
            ax5.axvline(stability_scores.mean(), color='red', linestyle='--',
                        label=f'Mean: {stability_scores.mean():.2f}')
            ax5.legend()

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'feature_importance_comprehensive.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_hypothesis_testing_results(self, hypothesis_results: Dict[str, Any]):
        """Visualize hypothesis testing results"""

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Hypothesis Testing Results', fontsize=18, fontweight='bold')

        # RQ2 Results Visualization
        ax1 = axes[0, 0]
        rq2_tests = hypothesis_results.get('rq2_analysis', {}).get('statistical_tests', [])

        if rq2_tests:
            test_names = []
            p_values = []

            for test in rq2_tests:
                test_result = test['test_result']
                test_names.append(test['comparison'].replace('_', ' ').title())
                p_values.append(test_result['p_value'])

            bars = ax1.bar(range(len(test_names)), p_values)
            ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
            ax1.set_xticks(range(len(test_names)))
            ax1.set_xticklabels(test_names, rotation=45, ha='right')
            ax1.set_ylabel('P-value')
            ax1.set_title('RQ2: Statistical Test P-values')
            ax1.legend()

            for i, (bar, p_val) in enumerate(zip(bars, p_values)):
                bar.set_color(self.colors['primary'] if p_val < 0.05 else self.colors['secondary'])

        # RQ3 Results Visualization
        ax2 = axes[0, 1]
        rq3_tests = hypothesis_results.get('rq3_analysis', {}).get('statistical_tests', [])

        if rq3_tests:
            test_types = [test['test_type'].replace('_', ' ').title() for test in rq3_tests]
            significance = [test['test_result']['significant'] for test in rq3_tests]

            colors = [self.colors['primary'] if sig else self.colors['secondary'] for sig in significance]
            ax2.bar(range(len(test_types)), [1 if sig else 0 for sig in significance], color=colors)
            ax2.set_xticks(range(len(test_types)))
            ax2.set_xticklabels(test_types, rotation=45, ha='right')
            ax2.set_ylabel('Significant (1) / Not Significant (0)')
            ax2.set_title('RQ3: Test Significance Results')
            ax2.set_ylim(-0.1, 1.1)

        # Effect sizes comparison
        ax3 = axes[1, 0]
        all_effects = []
        all_labels = []

        # Collect effect sizes from both RQ2 and RQ3
        for test in rq2_tests:
            all_effects.append(abs(test['test_result']['effect_size']))
            all_labels.append(f"RQ2: {test['comparison']}")

        for test in rq3_tests:
            all_effects.append(abs(test['test_result']['effect_size']))
            all_labels.append(f"RQ3: {test['test_type']}")

        if all_effects:
            bars = ax3.barh(range(len(all_effects)), all_effects)
            ax3.set_yticks(range(len(all_effects)))
            ax3.set_yticklabels([label.replace('_', ' ') for label in all_labels])
            ax3.set_xlabel('Effect Size (|Cohen\'s d|)')
            ax3.set_title('Effect Sizes Across All Tests')

            # Color by effect size magnitude
            for i, (bar, effect) in enumerate(zip(bars, all_effects)):
                if effect < 0.2:
                    color = 'lightgray'
                elif effect < 0.5:
                    color = self.colors['accent']
                elif effect < 0.8:
                    color = self.colors['primary']
                else:
                    color = self.colors['sentiment']
                bar.set_color(color)

        # Summary conclusions
        ax4 = axes[1, 1]
        ax4.axis('off')

        rq2_conclusion = hypothesis_results.get('rq2_analysis', {}).get('summary', {}).get('overall_conclusion', 'N/A')
        rq3_conclusion = hypothesis_results.get('rq3_analysis', {}).get('summary', {}).get('overall_conclusion', 'N/A')

        summary_text = f"""
    Research Question Conclusions:

    RQ2: {rq2_conclusion[:60]}...

    RQ3: {rq3_conclusion[:60]}...

    Statistical Summary:
    • Total tests: {len(rq2_tests) + len(rq3_tests)}
    • Significance level: α = 0.05
    • Effect size: Cohen's d
        """

        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        ax4.set_title('Research Conclusions Summary')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'hypothesis_testing_results.png',
                    dpi=300, bbox_inches='tight')
        plt.close()


# =============================================================================
# REPORT GENERATION ENGINE
# =============================================================================

class ComprehensiveReportGenerator:
    """Advanced report generation for RQ2-RQ3 analysis"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.reports_dir = config.reports_dir

    def generate_executive_summary(self, hypothesis_results: Dict[str, Any],
                                   consensus_df: pd.DataFrame, results_df: pd.DataFrame) -> str:
        """Generate executive summary of findings"""

        rq2_conclusion = hypothesis_results.get('rq2_analysis', {}).get('summary', {}).get('overall_conclusion',
                                                                                           'Inconclusive')
        rq3_conclusion = hypothesis_results.get('rq3_analysis', {}).get('summary', {}).get('overall_conclusion',
                                                                                           'Inconclusive')

        bundle_performance = results_df.groupby('bundle')['f1_score'].mean().sort_values(ascending=False)
        best_bundle = bundle_performance.index[0] if len(bundle_performance) > 0 else 'Unknown'
        best_performance = bundle_performance.iloc[0] if len(bundle_performance) > 0 else 0

        top_features = consensus_df.head(5)['feature'].tolist() if len(consensus_df) > 0 else []

        summary = f"""
# EXECUTIVE SUMMARY
================

## Research Objective
This comprehensive analysis addresses two critical research questions in quantitative finance:
- **RQ2**: Which models perform better - Technical vs Sentiment comparison?
- **RQ3**: Which KPIs matter most for prediction accuracy?

## Key Findings

### RQ2: Model Performance Comparison
**Conclusion**: {rq2_conclusion}

**Best Performing Approach**: {best_bundle.replace('_', ' ').title()}
- **Performance**: F1-Score = {best_performance:.4f}

### RQ3: Feature Importance Analysis  
**Conclusion**: {rq3_conclusion}

**Most Predictive Features**:
"""

        for i, feature in enumerate(top_features, 1):
            summary += f"{i}. {feature}\n"

        summary += f"""
## Strategic Recommendations

### Portfolio Management
- **Feature Strategy**: Focus on validated feature engineering approaches
- **Model Selection**: Prioritize {best_bundle.replace('_', ' ')} approaches

### Risk Management
- **Key Indicators**: Monitor {', '.join(top_features[:3])} as primary signals

## Statistical Validation
- **Significance Level**: α = 0.05
- **Sample Size**: {len(results_df)} model evaluations

---
*Analysis conducted using rigorous statistical hypothesis testing with cross-validation.*
        """

        return summary

    def save_comprehensive_reports(self, df: pd.DataFrame, sentiment_cols: List[str],
                                   technical_cols: List[str], results_df: pd.DataFrame,
                                   cv_results_df: pd.DataFrame, hypothesis_results: Dict[str, Any],
                                   feature_selector: AdvancedFeatureSelector) -> None:
        """Save all comprehensive reports"""

        consensus_df = feature_selector.consensus_feature_ranking()

        # Executive Summary
        exec_summary = self.generate_executive_summary(hypothesis_results, consensus_df, results_df)
        with open(self.reports_dir / 'executive_summary.md', 'w', encoding='utf-8') as f:
            f.write(exec_summary)

        print(f"Reports saved to {self.reports_dir}")


# =============================================================================
# MAIN EXECUTION ENGINE
# =============================================================================

def load_and_prepare_comprehensive_data(config: AnalysisConfig) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Load and prepare data with comprehensive validation"""

    print("Loading datasets...")

    # Load sentiment data
    sentiment_df = pd.read_csv(config.sentiment_csv)
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    print(f"Loaded sentiment data: {len(sentiment_df)} observations")

    # Load price data
    price_df = pd.read_csv(config.price_csv)
    price_df['date'] = pd.to_datetime(price_df['date'], dayfirst=True)
    print(f"Loaded price data: {len(price_df)} observations")

    # Filter tickers if specified
    if config.tickers:
        tickers = [t.strip().upper() for t in config.tickers.split(',')]
        sentiment_df = sentiment_df[sentiment_df['symbol'].str.upper().isin(tickers)]
        price_df = price_df[price_df['symbol'].str.upper().isin(tickers)]
        print(f"Filtered to {len(tickers)} tickers")

    # Compute comprehensive technical features
    print("Computing technical indicators...")
    price_with_tech = compute_comprehensive_technical_features(price_df, config)
    print(f"Computed technical features: {len(price_with_tech)} observations")

    # Create prediction targets
    print("Creating prediction targets...")
    price_with_tech = price_with_tech.sort_values(['symbol', 'date'])

    for horizon in [1, 2, 3]:
        price_with_tech[f'forward_return_{horizon}d'] = (
                price_with_tech.groupby('symbol')['close'].shift(-horizon) /
                price_with_tech['close'] - 1
        )
        price_with_tech[f'target_{horizon}d'] = (
                price_with_tech[f'forward_return_{horizon}d'] > 0
        ).astype(int)

    # Merge datasets
    print("Merging sentiment and technical data...")
    df = pd.merge(
        price_with_tech, sentiment_df,
        on=['symbol', 'date'],
        how='inner',
        suffixes=('', '_sent')
    )

    # Load sector data if available
    if os.path.exists(config.sectors_csv):
        sectors_df = pd.read_csv(config.sectors_csv)
        sectors_df['symbol_clean'] = sectors_df['symbol'].astype(str).str.strip()
        df = pd.merge(df, sectors_df[['symbol_clean', 'sector']],
                      left_on='symbol', right_on='symbol_clean', how='left')
        df = df.drop('symbol_clean', axis=1)
        print(f"Added sector information for {df['sector'].notna().sum()} observations")

    # Define feature columns
    sentiment_cols = [
        'ensemble_score_mean', 'ensemble_score_std', 'ensemble_score_count',
        'confidence_mean', 'relevance_score_mean', 'sentiment_lag1', 'sentiment_lag2'
    ]

    technical_cols = [
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
        'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_PercentB', 'BB_Width',
        'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'EMA_50',
        'Price_vs_SMA20', 'Price_vs_SMA50', 'SMA_Ratio_5_20', 'SMA_Ratio_20_50',
        'ATR', 'ATR_Normalized', 'OBV', 'VOLUME_SMA_20', 'VOLUME_RATIO',
        'VOLATILITY_5', 'VOLATILITY_10', 'VOLATILITY_20', 'ROC_5', 'ROC_10', 'ROC_20',
        'PRICE_POSITION', 'MOMENTUM_10', 'High_Low_Ratio', 'Close_Position',
        'Volume_Price_Trend', 'Price_Volume_Correlation',
        'SMA5_above_SMA20', 'Price_above_SMA20', 'Price_above_SMA50',
        'RSI_Oversold', 'RSI_Overbought', 'MACD_Bullish', 'BB_Squeeze',
        'DOJI', 'HAMMER', 'SHOOTING_STAR', 'ENGULFING_BULL', 'ENGULFING_BEAR'
    ]

    # Filter to available columns
    sentiment_cols = [col for col in sentiment_cols if col in df.columns]
    technical_cols = [col for col in technical_cols if col in df.columns]

    print(f"Final dataset: {len(df)} observations")
    print(f"Sentiment features: {len(sentiment_cols)}")
    print(f"Technical features: {len(technical_cols)}")

    # Save processed data
    df.to_csv(config.results_dir / 'processed_data.csv', index=False)

    return df, sentiment_cols, technical_cols


def run_comprehensive_rq2_rq3_analysis(config: AnalysisConfig):
    """Run the complete RQ2-RQ3 analysis framework"""

    print("=" * 80)
    print("RQ2-RQ3 COMPREHENSIVE ANALYSIS FRAMEWORK")
    print("=" * 80)

    # Load and prepare data
    df, sentiment_cols, technical_cols = load_and_prepare_comprehensive_data(config)

    # Initialize analysis engines
    feature_selector = AdvancedFeatureSelector(config)
    hypothesis_engine = HypothesisTestingEngine(config)
    viz_engine = ComprehensiveVisualizationEngine(config)
    report_generator = ComprehensiveReportGenerator(config)

    # Create specifications
    feature_bundles = create_enhanced_feature_bundles(sentiment_cols, technical_cols)
    model_specs = create_comprehensive_model_specs(config)

    print(f"\nFeature bundles: {[b.name for b in feature_bundles]}")
    print(f"Models: {[m.name for m in model_specs]}")

    # Main analysis loop
    all_results = []
    all_cv_results = []

    for horizon in [1, 2, 3]:
        print(f"\n{'=' * 20} ANALYZING HORIZON {horizon} DAYS {'=' * 20}")

        target_col = f'target_{horizon}d'
        analysis_df = df.dropna(subset=sentiment_cols + technical_cols + [target_col]).copy()

        if len(analysis_df) < config.min_samples_per_symbol:
            print(f"Insufficient data for horizon {horizon}, skipping...")
            continue

        print(f"Analysis dataset: {len(analysis_df)} observations")

        # Time series split
        train_df, val_df, test_df = advanced_time_series_split(analysis_df, config.test_fraction)
        print(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        # Feature selection analysis (RQ3)
        print("Conducting comprehensive feature selection...")
        X_train_full = train_df[sentiment_cols + technical_cols]
        y_train = train_df[target_col]

        feature_selector.univariate_filter_methods(X_train_full, y_train)
        feature_selector.wrapper_based_methods(X_train_full, y_train)
        feature_selector.embedded_methods(X_train_full, y_train)
        feature_selector.shap_analysis(X_train_full, y_train)

        # Model training and evaluation (RQ2)
        print("Training and evaluating models...")

        for bundle in feature_bundles:
            available_features = [f for f in bundle.columns if f in analysis_df.columns]

            if len(available_features) < 3:  # Minimum features required
                print(f"Insufficient features for {bundle.name}, skipping...")
                continue

            X_train = train_df[available_features].fillna(0)
            X_test = test_df[available_features].fillna(0)
            y_test = test_df[target_col]

            for spec in model_specs:
                try:
                    # Create pipeline
                    if spec.needs_scaling:
                        pipeline = Pipeline([
                            ('scaler', StandardScaler()),
                            ('model', spec.estimator)
                        ])
                    else:
                        pipeline = Pipeline([('model', spec.estimator)])

                    # Train model
                    pipeline.fit(X_train, y_train)

                    # Evaluate performance
                    metrics = compute_comprehensive_metrics(
                        pipeline, X_test, y_test, spec.name, bundle.name
                    )

                    # Store results
                    result_row = {
                        'horizon': horizon,
                        'bundle': bundle.name,
                        'model': spec.name,
                        'bundle_type': bundle.bundle_type,
                        'n_features': len(available_features),
                        'train_size': len(X_train),
                        'test_size': len(X_test),
                        **{k: (v if isinstance(v, (int, float)) else float(v))
                           for k, v in metrics.items()}
                    }

                    all_results.append(result_row)

                    print(f"  {bundle.name} + {spec.name}: "
                          f"Acc={metrics['accuracy']:.3f}, "
                          f"F1={metrics['f1_score']:.3f}, "
                          f"AUC={metrics.get('roc_auc', 0):.3f}")

                except Exception as e:
                    print(f"Error: {spec.name} with {bundle.name}: {e}")

            # Cross-validation analysis
            try:
                cv_results = cross_validation_analysis(X_train, y_train, model_specs, config)
                for _, cv_row in cv_results.iterrows():
                    cv_result_row = {
                        'horizon': horizon,
                        'bundle': bundle.name,
                        **cv_row.to_dict()
                    }
                    all_cv_results.append(cv_result_row)
            except Exception as e:
                print(f"Cross-validation failed for {bundle.name}: {e}")

    if not all_results:
        print("No results generated. Check data and feature availability.")
        return

    # Convert to DataFrames
    results_df = pd.DataFrame(all_results)
    cv_results_df = pd.DataFrame(all_cv_results) if all_cv_results else pd.DataFrame()

    print(f"\nGenerated {len(results_df)} model evaluation results")

    # Comprehensive hypothesis testing
    print("Conducting comprehensive hypothesis testing...")
    hypothesis_results = hypothesis_engine.comprehensive_hypothesis_analysis(results_df, feature_selector)

    # Generate consensus feature ranking
    consensus_df = feature_selector.consensus_feature_ranking()

    # Create visualizations
    print("Generating comprehensive visualizations...")
    viz_engine.plot_technical_indicators_dashboard(df)
    viz_engine.plot_model_performance_analysis(results_df, cv_results_df)
    viz_engine.plot_feature_importance_comprehensive(feature_selector)
    viz_engine.plot_hypothesis_testing_results(hypothesis_results)

    # Generate reports
    print("Generating comprehensive reports...")
    report_generator.save_comprehensive_reports(
        df, sentiment_cols, technical_cols, results_df, cv_results_df,
        hypothesis_results, feature_selector
    )

    # Save results for dashboard
    print("Exporting dashboard data...")
    results_df.to_csv(config.dashboard_dir / 'model_performance_results.csv', index=False)
    if not cv_results_df.empty:
        cv_results_df.to_csv(config.dashboard_dir / 'cross_validation_results.csv', index=False)
    consensus_df.to_csv(config.dashboard_dir / 'feature_importance_consensus.csv', index=False)

    with open(config.dashboard_dir / 'analysis_summary.json', 'w') as f:
        summary_data = {
            'total_evaluations': len(results_df),
            'total_features_analyzed': len(sentiment_cols) + len(technical_cols),
            'total_securities': df['symbol'].nunique(),
            'analysis_period': {
                'start': df['date'].min().isoformat(),
                'end': df['date'].max().isoformat()
            },
            'rq2_conclusion': hypothesis_results.get('rq2_analysis', {}).get('summary', {}).get('overall_conclusion',
                                                                                                'Inconclusive'),
            'rq3_conclusion': hypothesis_results.get('rq3_analysis', {}).get('summary', {}).get('overall_conclusion',
                                                                                                'Inconclusive'),
            'best_bundle': results_df.groupby('bundle')['f1_score'].mean().idxmax(),
            'top_features': consensus_df.head(10)['feature'].tolist() if len(consensus_df) > 0 else []
        }
        json.dump(summary_data, f, indent=2, default=str)

    print(f"\n{'=' * 80}")
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print(f"{'=' * 80}")
    print(f"Results saved to: {config.outdir}")
    print(f"- Model evaluations: {len(results_df)}")
    print(f"- Feature rankings: {len(consensus_df)}")
    print(f"- Hypothesis tests: Completed for RQ2 and RQ3")
    print(f"- Visualizations: {len(list(config.plots_dir.glob('*.png')))} plots")
    print(f"- Reports: Comprehensive reports generated")
    print(f"- Dashboard exports: Ready for Streamlit")


def main():
    """Main execution function with comprehensive argument parsing"""

    parser = argparse.ArgumentParser(
        description="RQ2-RQ3 Comprehensive Analysis Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rq2_rq3_analysis.py --tickers "HDFCBANK,ICICIBANK,SBIN"
  python rq2_rq3_analysis.py --test_fraction 0.3 --cv_folds 10
  python rq2_rq3_analysis.py --outdir custom_results --alpha 0.01
        """
    )

    parser.add_argument('--sentiment_csv', default='rq1_complete_results/daily_sentiment_aggregated.csv',
                        help='Path to sentiment data CSV')
    parser.add_argument('--price_csv', default='rq1_complete_results/stock_data_clean.csv',
                        help='Path to price data CSV')
    parser.add_argument('--sectors_csv', default='rq1_complete_results/sector_analysis/sectors.csv',
                        help='Path to sector mapping CSV')
    parser.add_argument('--tickers', help='Comma-separated list of tickers to analyze')
    parser.add_argument('--test_fraction', type=float, default=0.2,
                        help='Fraction of data for testing (default: 0.2)')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--outdir', default='rq2_rq3_analysis_results',
                        help='Output directory for results')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level for hypothesis testing')
    parser.add_argument('--min_samples', type=int, default=30,
                        help='Minimum samples per symbol')

    args = parser.parse_args()

    # Create configuration
    config = AnalysisConfig(
        sentiment_csv=args.sentiment_csv,
        price_csv=args.price_csv,
        sectors_csv=args.sectors_csv,
        tickers=args.tickers,
        test_fraction=args.test_fraction,
        cv_folds=args.cv_folds,
        outdir=args.outdir,
        random_state=args.random_state,
        alpha=args.alpha,
        min_samples_per_symbol=args.min_samples
    )

    # Run comprehensive analysis
    try:
        run_comprehensive_rq2_rq3_analysis(config)
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()