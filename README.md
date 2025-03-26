# Financial Market Analysis: Volatility Modeling and Comparison with Hedging Instruments
author: František Šuráň
email: frantasuran26@gmail.com

This application provides tools for analyzing financial market data, focusing on volatility modeling and correlation analysis between stocks and hedging instruments with implementation of GARCH models.

## Features:

### Data Fetching
Historical stocks price data retrieval from Polygon.io API, also data for either gold or silver Configurable lookback period (30-180 days) - due to the limitations of API API key hidden in the .env file

### Analysis Modules Correlation Analysis: 
Dual-axis price comparison plots Log returns calculation and visualization Rolling correlation analysis

### GARCH Volatility Analysis: 
Automated GARCH parameter selection (p,q) Model comparison and selestion (GARCH vs GJR-GARCH) Visualization of the conditional volatility

## Running the application:
First download the required packages:
pip install -r requirements.txt

One can start the application by running the following command in the terminal:

First, create the virtual environemnt python3 -m venv bipyt
Activate the virtual environment: source bipyt/bin/activate (in windows bipyt\\Scripts\\activate)
Download the necessary packages pip install jupyter numpy pandas matplotlib pillow pytest together with the additioal requirements.txt

streamlit run FINMARKETAPP/main.py

Then to properly use it:

Enter a stock ticker in the sidebar - if wrongly input, one can check the available tickers (you can try to input not existing ticker) Select the desired lookback period (30-180 days) Choose a hedging instrument (Gold or Silver) Click "Analyze" to generate. Then one can switch between the generated tabs.

To run all tests one can write the following command into the terminal:

pytest tests/