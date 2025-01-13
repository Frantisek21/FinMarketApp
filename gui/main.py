import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent)) # ChatGPT
import streamlit as st
from FinMarketApp.src.question.text_input import DataFetcher
from FinMarketApp.src.question.multi_choice import CorrelationAnalysis, GarchAnalysis


def main():
    """Main function for the Financial Market Analysis app."""
    st.title(
        "Financial Market Analysis: Volatility Modeling and "
        "Comparison with Hedging Instruments"
    )

    # Initialize the data fetcher
    data_fetcher = DataFetcher()

    # Sidebar for inputs
    st.sidebar.header("Parameters")
    user_ticker = st.sidebar.text_input(
        "Enter Stock Ticker:"
    ).strip().upper()
    days_lookback = st.sidebar.slider(
        "Days of historical data:", 30, 180, 180
    )

    # Add radio buttons for selecting hedging instrument
    hedging_instrument = st.sidebar.radio(
        "Select Hedging Instrument:", options=["Gold", "Silver"]
    )

    # Map selection to ticker symbol and color
    hedge_ticker = (
        "C:XAUUSD" if hedging_instrument == "Gold" else "C:XAGUSD"
    )
    hedge_color = "orange" if hedging_instrument == "Gold" else "gray"

    if st.sidebar.button("Analyze"):
        # Define tickers (user's stock and selected hedge)
        tickers = [user_ticker, hedge_ticker]

        # Fetch data
        dataframes = data_fetcher.get_data(tickers, days=days_lookback)

        if dataframes and len(dataframes) == 2:
            # Initialize Analysis Classes
            correlation_analyzer = CorrelationAnalysis()
            garch_analyzer = GarchAnalysis()

            # Display tabs for different analyses
            tab1, tab2, tab3 = st.tabs(
                ["Price Data", "Correlation Analysis", "Volatility Analysis"]
            )

            with tab1:
                st.subheader("Price Data")
                st.write(f"""
                **Description:**
                This tab displays the historical prices of **{user_ticker}**
                and **{hedging_instrument}**
                over the past **{days_lookback} days**.
                It provides a clear comparison of how both assets performed,
                helping to visually assess trends and patterns.
                """)
                correlation_analyzer.plot_dual_axis(
                    dataframes, user_ticker, hedge_ticker, hedge_color
                )

            with tab2:
                st.subheader(f"Correlation with {hedging_instrument}")
                st.write(f"""
                **Description:**
                This tab explores the **log returns** of **{user_ticker}**
                and **{hedging_instrument}** over
                the past **{days_lookback} days**.
                - **Log Returns Plot:** Shows how the returns fluctuate over
                  time, giving insights into volatility.
                - **Rolling Correlation:** Measures the dynamic relationship
                  between the two assets over a rolling window.
                  - A **positive correlation** indicates the assets move
                    together, while a **negative correlation** suggests they
                    move in opposite directions.
                  - **Note:** Rolling correlation results often vary and may
                    not provide a definitive relationship between
                    **{user_ticker}** and **{hedging_instrument}**. It serves
                    as a general indicator rather than a concrete conclusion.
                """)
                correlation_analyzer.analyze_correlation(
                    dataframes, user_ticker, hedge_ticker, hedge_color
                )

            with tab3:
                st.subheader("GARCH Volatility Analysis")
                st.write(f"""
                **Description:**
                This tab focuses on **GARCH-based volatility modeling** for
                **{user_ticker}**, using historical data from the past
                **{days_lookback} days**. The analysis provides insights into
                the stock's volatility dynamics, helping to evaluate the
                **Volatility Clustering** and understanding how volatility
                evolves over time in response to market conditions.
                The value of p reflects the extent to which the model relies on
                past shocks or sudden market changes to predict current
                volatility, while q indicates the degree to which past
                volatility levels affect the present. A model with lower
                values of p and q (such as GARCH(1,1)) places greater emphasis
                on the most recent market movements.
                """)
                garch_analyzer.analyze_volatility(dataframes[user_ticker])
        else:
            # Show the error message with a clickable link
            st.error(
                "Unable to fetch data for all required tickers. "
                "Please ensure your inputs are correct and available at "
                "[Polygon.io Tickers](https://polygon.io/quote/tickers)."
            )
            st.info(
                "Please avoid querying the same ticker more than "
                "once per minute to prevent API rate-limiting issues."
            )
    else:
        # Default info message before analysis
        st.info(
            "Please avoid querying the same ticker more than once per minute "
            "to prevent API rate-limiting issues."
        )


if __name__ == "__main__":
    main()
