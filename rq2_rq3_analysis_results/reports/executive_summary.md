
# EXECUTIVE SUMMARY
================

## Research Objective
This comprehensive analysis addresses two critical research questions in quantitative finance:
- **RQ2**: Which models perform better - Technical vs Sentiment comparison?
- **RQ3**: Which KPIs matter most for prediction accuracy?

## Key Findings

### RQ2: Model Performance Comparison
**Conclusion**: Evidence against H1: Individual models may perform as well as or better than hybrid models

**Best Performing Approach**: Technical Volume
- **Performance**: F1-Score = 0.5213

### RQ3: Feature Importance Analysis  
**Conclusion**: Strong support for H1: Significant variation in feature importance with identifiable dominant KPIs

**Most Predictive Features**:
1. BB_Width
2. ensemble_score_std
3. OBV
4. ensemble_score_mean
5. SMA_Ratio_5_20

## Strategic Recommendations

### Portfolio Management
- **Feature Strategy**: Focus on validated feature engineering approaches
- **Model Selection**: Prioritize technical volume approaches

### Risk Management
- **Key Indicators**: Monitor BB_Width, ensemble_score_std, OBV as primary signals

## Statistical Validation
- **Significance Level**: α = 0.05
- **Sample Size**: 168 model evaluations

---
*Analysis conducted using rigorous statistical hypothesis testing with cross-validation.*
        