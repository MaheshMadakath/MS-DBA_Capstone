## Data Description

    ### Dataset Overview
    The research dataset comprises stock price data, news articles, and derived sentiment scores for **26 NIFTY-listed companies** spanning **2025-02-01 00:00:00 to 2025-08-14 00:00:00**.

    ### Stock Price Data
    - **Source**: Yahoo Finance via yfinance Python library
    - **Coverage**: 26 stocks across major sectors (Banking, IT, Energy, FMCG, Automotive, Defense, Jewelry)
    - **Variables**: Open, High, Low, Close prices and Volume
    - **Derived Features**: 1-day, 2-day, and 3-day returns; rolling volatility
    - **Total Observations**: 3,458 stock-day combinations

    **Stocks Analyzed**:
    BEL.NS, BHEL.NS, BPCL.NS, COCHINSHIP.NS, GRSE.NS, HAL.NS, HCLTECH.NS, HDFCBANK.NS, HINDUNILVR.NS, ICICIBANK.NS, INFY.NS, ITC.NS, KALYANKJIL.NS, KOTAKBANK.NS, M&M.NS, MARUTI.NS, NESTLEIND.NS, ONGC.NS, RAJESHEXPO.NS, RELIANCE.NS, SBIN.NS, TATAMOTORS.NS, TCS.NS, THANGAMAYL.NS, TITAN.NS, WIPRO.NS

    ### News Data Collection
    - **Total Articles**: 3,528
    - **Average Relevance Score**: 79.8%
    - **Collection Method**: Multi-source approach combining MoneyControl stock-specific pages and Google News RSS feeds

    **Source Distribution**:
- GoogleNews_business-standard: 482 articles (13.7%)
- GoogleNews_cnbctv18: 375 articles (10.6%)
- GoogleNews_economictimes: 400 articles (11.3%)
- GoogleNews_livemint: 619 articles (17.5%)
- GoogleNews_zeebiz: 286 articles (8.1%)
- MoneyControl_Local: 1,366 articles (38.7%)

    ### Sentiment Analysis
    - **Method**: Ensemble approach combining three models
      - FinBERT (50% weight): Domain-specific financial sentiment model
      - VADER (30% weight): Rule-based social media sentiment analyzer
      - TextBlob (20% weight): General-purpose sentiment classifier
    - **Aggregation**: Daily sentiment scores per stock with confidence measures
    - **Features**: Current sentiment, 1-day lag, 2-day lag, confidence scores

    ### Data Quality Metrics
    - **Temporal Coverage**: 194 trading days
    - **Missing Data**: Minimal (<5% for key variables after preprocessing)
    - **Relevance Validation**: 79.8% average relevance score exceeds 70% target
    - **Sample Size**: 1,138 sentiment-return pairs for analysis

    ### Data Accessibility
    All datasets are publicly accessible and reproducible:
    - **Stock Data**: Available through Yahoo Finance API
    - **News Sources**: Publicly accessible financial news websites
    - **Sentiment Models**: Open-source implementations
    - **GitHub Repository**: [Will be provided upon completion]

    The dataset design ensures sufficient statistical power for correlation analysis and machine learning model validation, meeting the sample size requirements established in the research methodology.