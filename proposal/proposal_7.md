
# Capstone Proposal
## Policy Driven Energy Price Shocks and Macroeconomic Transmission
### Proposed by: Aswin Balaji Thippa Ramesh, Vishal Fulsundar
#### Email: at119@gwu.edu, vishal.fulsundar@gwu.edu
#### Advisor: Amir Jafari
#### The George Washington University, Washington DC  
#### Data Science Program


## 1 Objective:  

    This capstone project will develop a multimodal forecasting and analysis framework that studies how policy-driven
    events and economic news trigger energy price shocks, and how those shocks transmit into macroeconomic outcomes
    such as inflation, interest rates, and output.

    The system integrates:
    (1) Energy price time series (oil/gas),
    (2) Policy and economic news text,
    (3) Macroeconomic indicators (GDP, CPI, rates).

    It uses dual modeling pathways:
    - Time Series Branch: learns energy price dynamics and detects shock patterns (level shifts, volatility spikes).
    - NLP Branch: extracts policy and economic signals from news (sentiment, event type, topic shifts).

    A Fusion Layer combines signals from both branches to produce:
    - Energy price shock likelihood forecasts
    - Expected macroeconomic impact forecasts
    - Evidence-backed explanations grounded in retrieved news/policy context

    The outcome is an applied, research-oriented system that improves interpretability and robustness of energy price
    modeling by incorporating policy/news drivers and macroeconomic transmission pathways.
    

![Figure 1: Example figure](2026_Spring_7.png)
*Figure 1: Caption*

## 2 Dataset:  

    The project will rely only on publicly available datasets and sources.

    DATASET 1 — Energy Price Time Series:
    - Crude Oil (WTI), Natural Gas (Henry Hub), Electricity price series (optional)
    - Source: U.S. Energy Information Administration (EIA) or equivalent public series
    - Stored fields: date/time, price level, returns, volatility measures, shock labels

    DATASET 2 — News Text (Policy & Economic News):
    - Policy news: sanctions, regulations, government announcements, SPR releases
    - Economic news: inflation narratives, rate decisions, macro releases
    - Sources: GDELT (global event/news), RSS feeds from reputable outlets, government press releases (DOE/Treasury/Fed)
    - Stored fields: timestamp, headline, article snippet/full text, publisher, extracted topics, sentiment, event tags

    DATASET 3 — Economic Factor Data:
    - GDP, CPI, unemployment, industrial production, interest rates (Fed funds)
    - Sources: FRED (Federal Reserve Economic Data), OECD (optional)
    - Stored fields: date, macro series values, derived regime labels (optional)

    NOTE:
    - No energy demand dataset used.
    - No climate/weather dataset used.
    

## 3 Rationale:  

    Energy prices are highly sensitive to policy shocks (sanctions, regulations, geopolitical decisions) and macroeconomic
    narratives. These shocks often propagate into the broader economy by raising production and transportation costs,
    increasing inflation pressure, and influencing monetary policy decisions.

    Traditional forecasting methods often fail to explain *why* a shock occurred and struggle during structural breaks.
    This project improves both forecasting and interpretability by combining time-series modeling with NLP-based policy/news
    signal extraction and a fusion framework to link policy triggers to price shocks and macroeconomic transmission.
    

## 4 Approach:  

    The project will be implemented in the following stages:

    1) Data Collection & ETL
       - Pull energy price series (EIA/FRED where applicable)
       - Collect policy and economic news text (GDELT + RSS + official press releases)
       - Pull macroeconomic factor series (FRED)
       - Standardize timestamps and build unified dataset index

    2) Preprocessing & Feature Engineering
       - Energy prices: returns, rolling volatility, shock detection labels (spikes, breakpoints)
       - News text: cleaning, tokenization, embeddings, event tagging (sanctions/regulations/rates/inflation)
       - Macro data: alignment to price frequency; optional regime labels

    3) Dual Processing Pathways
       A) Time Series Branch (Energy Prices)
          - Models: ARIMA/Prophet baseline + LSTM/GRU/TCN/Transformer (advanced)
          - Outputs: shock probability, volatility forecast, next-step price movement

       B) NLP Branch (Policy & Economic News)
          - Models: FinBERT/RoBERTa sentiment + topic clustering + event classification
          - Outputs: policy shock score, macro narrative score, event type probability

    4) Fusion Layer
       - Combine time series embeddings + NLP embeddings/features (early/late fusion)
       - Learn weighting mechanism for news-driven vs. market-driven movements
       - Produce unified predictions with confidence scoring

    5) Prediction Output + Explanation
       - Energy price shock likelihood (short horizon)
       - Expected macro impact (inflation pressure, rate response proxy, output proxy)
       - Explanation: top news drivers + linked timestamps + retrieved evidence snippets (optional RAG grounding)

    6) Evaluation
       - Forecast metrics: MAE, RMSE, MAPE, directional accuracy
       - Shock detection: precision/recall on labeled shock periods
       - Ablations: time-series only vs. NLP only vs. fused model
       - Explainability: qualitative checks + attribution (attention/SHAP if used)
    

## 5 Timeline:  

    Week 1:
      - Finalize scope: energy series (WTI/gas/elec) + macro series (CPI/GDP/rates) + news source plan
      - Define evaluation metrics (shock detection + macro transmission)

    Week 2:
      - Implement collectors: energy prices + macro factors
      - Implement news ingestion (GDELT/RSS) and storage schema

    Week 3:
      - Build preprocessing pipeline: returns/volatility + news cleaning + timestamp alignment
      - Create baseline shock labeling rules

    Week 4:
      - Baseline models: ARIMA/Prophet for prices + simple sentiment baseline
      - Initial EDA + baseline performance report

    Week 5:
      - Train time-series deep model (LSTM/TCN/Transformer)
      - Add volatility/spike modeling improvements

    Week 6:
      - Train NLP models (FinBERT/RoBERTa) for sentiment + topic/event extraction
      - Build policy/economic event tagging logic

    Week 7:
      - Implement fusion layer (feature concatenation / attention fusion)
      - Compare fused vs. non-fused

    Week 8:
      - Add confidence scoring + explanation outputs (top drivers, attribution)
      - Improve data quality + robustness

    Week 9:
      - Expand news coverage + improve event classification
      - Add optional RAG grounding for key claims (retrieved snippets)

    Week 10:
      - Full backtesting pipeline + ablation studies
      - Stress test on historical shock events

    Week 11:
      - Evaluation report writing + charts + tables
      - Model selection + finalize best pipeline

    Week 12:
      - Build Streamlit demo: inputs → predictions + explanations
      - Run end-to-end demo tests

    Week 13:
      - Final documentation + reproducibility + packaging
      - Slide deck draft and final checks

    Week 14:
      - Final capstone submission + presentation + demo

    TOTAL: 14 weeks
    


## 6 Expected Number Students:  

    RECOMMENDED: 1–2 students

    If 2 students:
    Student 1:
      - Data ingestion + ETL + time-series modeling + evaluation
    Student 2:
      - NLP pipeline + fusion layer + explanation methods + demo UI
    

## 7 Possible Issues:  

    TECHNICAL CHALLENGES AND SOLUTIONS:
    - News noise and relevance:
      * Mitigation: filtering keywords, event classification, source quality ranking
    - Time alignment across frequencies (daily news vs monthly GDP):
      * Mitigation: resampling rules + lag features + monthly-to-daily mapping where needed
    - Shock labeling quality:
      * Mitigation: combine rule-based + change-point detection; validate on known shock periods
    - Model overfitting:
      * Mitigation: backtesting, walk-forward validation, strong baselines, ablations
    


## Contact
- Author: Amir Jafari
- Email: [ajafari@gwu.edu](mailto:ajafari@gwu.edu)
- GitHub: [](https://github.com/)
