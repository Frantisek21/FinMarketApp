# Financial Market Analysis: Volatility Modeling and Comparison with Hedging Instruments
author: František Šuráň
email: frantasuran26@gmail.com

This application provides tools for analyzing financial market data, focusing on volatility modeling and correlation analysis between stocks and hedging instruments with implementation of GARCH models.

For the creation of the structure and with the help of understanding it was used chatGPT, where I was experimenting with several possible categories. As I had no experience with this, I decided to use the proposed structure from the How to Semestrálka (nejen pro BI-PYT), therefore the naming is the same.

app/
├── src/
│   ├── base.py              # Base class for API interactions
│   ├── question/
│   │   ├── text_input.py    # Data fetching functionality
│   │   └── multi_choice.py  # Analysis modules
├── tests/
│   ├── test_analysis.py     # Tests for GARCH analysis
│   ├── test_data.py         # Tests for data fetching
│   └── test_main.py         # Integration tests
└── gui/main.py              # Streamlit application entry point

(plus "__init__.py" files)

Features:
1. Data Fetching

Historical stocks price data retrieval from Polygon.io API, also data for either gold or silver
Configurable lookback period (30-180 days) - due to the limitations of API
API key hidden in the .env file

2. Analysis Modules
Correlation Analysis:
Dual-axis price comparison plots
Log returns calculation and visualization
Rolling correlation analysis

GARCH Volatility Analysis:
Automated GARCH parameter selection (p,q)
Model comparison and selestion (GARCH vs GJR-GARCH)
Visualization of the conditional volatility

Running the application:

One can start the application by running the following command in the terminal:

streamlit run app/main.py

Then to properly use it:

Enter a stock ticker in the sidebar - if wrongly input, one can check the available tickers (you can try to input not existing ticker)
Select the desired lookback period (30-180 days)
Choose a hedging instrument (Gold or Silver)
Click "Analyze" to generate. Then one can switch between the generated tabs.

To run all tests one can write the following command into the terminal:

pytest app/tests/

