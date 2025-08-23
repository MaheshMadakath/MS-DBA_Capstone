# Research Question 1 Analysis Report
## News Sentiment Predictive Power on Stock Returns

### Executive Summary
This analysis investigates the short-term (1-3 day) predictive power of news sentiment on stock price changes across sectors using an ensemble sentiment analysis approach.

### Methodology
- **Sentiment Analysis**: FinBERT (50%) + VADER (30%) + TextBlob (20%) ensemble
- **Data Sources**: MoneyControl stock-specific URLs + Google News RSS feeds
- **Statistical Methods**: Correlation analysis, regression modeling, classification analysis
- **Machine Learning**: Random Forest and Logistic Regression models

### Key Findings

#### Correlation Analysis

**Pearson Correlations (Sentiment vs Returns):**
- 1-Day: -0.1396 (p-value: 0.0000)
- 2-Day: -0.1104 (p-value: 0.0006)
- 3-Day: -0.1149 (p-value: 0.0003)

#### Regression Analysis (Random Forest)

**Returns_1d:**
- R² Score: -0.1466
- MSE: 0.000701

**Returns_2d:**
- R² Score: -0.1825
- MSE: 0.000934

**Returns_3d:**
- R² Score: -0.1845
- MSE: 0.001203

#### Classification Analysis (Direction Prediction)

**Returns_1d_binary - Logistic Regression:**
- Accuracy: 0.4973
- Precision: 0.5000
- Recall: 0.5161
- F1-Score: 0.5079

**Returns_1d_binary - Random Forest:**
- Accuracy: 0.5135
- Precision: 0.5152
- Recall: 0.5484
- F1-Score: 0.5312

**Returns_2d_binary - Logistic Regression:**
- Accuracy: 0.5514
- Precision: 0.5526
- Recall: 0.4615
- F1-Score: 0.5030

**Returns_2d_binary - Random Forest:**
- Accuracy: 0.6000
- Precision: 0.5810
- Recall: 0.6703
- F1-Score: 0.6224

**Returns_3d_binary - Logistic Regression:**
- Accuracy: 0.5838
- Precision: 0.6207
- Recall: 0.3956
- F1-Score: 0.4832

**Returns_3d_binary - Random Forest:**
- Accuracy: 0.6054
- Precision: 0.5978
- Recall: 0.6044
- F1-Score: 0.6011

#### Statistical Significance Tests

**T-Test (Positive vs Negative Sentiment Returns):**
- T-statistic: nan
- P-value: nan
- Significant: False

**ANOVA (Sentiment Categories):**
- F-statistic: nan
- P-value: nan
- Significant: False

### Conclusions
[Add your research conclusions based on the statistical results above]

### Limitations and Future Work
1. Sample size and time period constraints
2. Market regime dependency
3. Sector-specific variations
4. External factor influences

### Data Quality Metrics
- Target relevance rate achieved: 60-80%
- News articles per stock: 10-15 (target met)
- Statistical significance: [Based on p-values above]

---
*Report generated on 2025-08-22 01:26:50*
