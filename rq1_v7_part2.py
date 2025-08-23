from presentation_builder_v7 import build_presentation_json

FOCUS_SECTORS = {
    'Banking': {'stocks': ['HDFCBANK.NS','ICICIBANK.NS','SBIN.NS','KOTAKBANK.NS'], 'market_impact':'High','volatility':'Medium','expected_growth':12.0},
    'IT_Services': {'stocks': ['TCS.NS','INFY.NS','HCLTECH.NS','WIPRO.NS'], 'market_impact':'High','volatility':'High','expected_growth':8.5},
    'Energy': {'stocks': ['RELIANCE.NS','ONGC.NS','BPCL.NS'], 'market_impact':'Very High','volatility':'High','expected_growth':7.5},
    'FMCG': {'stocks': ['ITC.NS','HINDUNILVR.NS','NESTLEIND.NS'], 'market_impact':'Medium','volatility':'Low','expected_growth':9.0},
    'Automotive': {'stocks': ['MARUTI.NS','M&M.NS','TATAMOTORS.NS'], 'market_impact':'Medium','volatility':'Very High','expected_growth':6.0},
    'Defense': {'stocks': ['HAL.NS','BEL.NS','BHEL.NS','COCHINSHIP.NS','GRSE.NS'], 'market_impact':'Low','volatility':'Extreme','expected_growth':25.0},
    'Jewelry': {'stocks': ['KALYANKJIL.NS','TITAN.NS','RAJESHEXPO.NS','THANGAMAYL.NS'], 'market_impact':'Medium','volatility':'High','expected_growth':15.0}
}

out_path = build_presentation_json(
    "rq1_enhanced_results",
    combined_json_name="streamlit_rq1_v7_presentation.json",
    focus_sectors=FOCUS_SECTORS,  # you can omit this if you don't want sector info
    # sector_map=..., sector_meta=...,           # optional overrides
    analysis_period="6mo",
)
print("✅ Presentation JSON:", out_path)
