import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent)) # ChatGPT
from app.src.base import BaseAnalysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera
from arch import arch_model
import streamlit as st


class CorrelationAnalysis(BaseAnalysis):
    def __init__(self):
        super().__init__()

    def plot_dual_axis(
            self, dataframes, user_ticker, hedge_ticker, hedge_color
            ):
        """
        Plot closing prices for the specified user ticker and hedging
        instrument with dual y-axes.
        """
        if user_ticker not in dataframes or dataframes[user_ticker].empty:
            st.error(f"No data to plot for {user_ticker}.")
            return
        if hedge_ticker not in dataframes or dataframes[hedge_ticker].empty:
            st.error(f"No data to plot for {hedge_ticker}.")
            return

        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Plot user ticker data on the left y-axis
        ax1.plot(dataframes[user_ticker].index,
                 dataframes[user_ticker]['Close'],
                 label=user_ticker, color='blue')
        ax1.set_ylabel(f"{user_ticker} Close Price",
                       color='blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_xlabel("Date", fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Create a second y-axis for the hedging instrument data
        ax2 = ax1.twinx()
        ax2.plot(dataframes[hedge_ticker].index,
                 dataframes[hedge_ticker]['Close'],
                 label=hedge_ticker, color=hedge_color)

        ax2.set_ylabel(f"{hedge_ticker} Close Price",
                       color=hedge_color, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=hedge_color)

        # Add title and legend
        plt.title(f"Closing Prices: {user_ticker} vs {hedge_ticker}",
                  fontsize=16)
        fig.tight_layout()

        st.pyplot(fig)

    def analyze_correlation(self,
                            dataframes,
                            user_ticker,
                            hedge_ticker,
                            hedge_color,
                            rolling_window=30):
        """
        Analyze the correlation between a user's stock
        ticker and hedging instrument.
        """
        if user_ticker not in dataframes or dataframes[user_ticker].empty:
            st.error(f"No data available for {user_ticker}.")
            return

        if hedge_ticker not in dataframes or dataframes[hedge_ticker].empty:
            st.error(f"No data available for {hedge_ticker}.")
            return

        # Align Timestamps
        aligned_data = pd.merge(
            dataframes[user_ticker],
            dataframes[hedge_ticker],
            left_index=True,
            right_index=True,
            suffixes=(f'_{user_ticker}', f'_{hedge_ticker}')
        )

        # Compute Log Returns
        aligned_data[f'Log_Returns_{user_ticker}'] = np.log(
            aligned_data[f'Close_{user_ticker}']
            / aligned_data[f'Close_{user_ticker}'].shift(1)
        )
        aligned_data[f'Log_Returns_{hedge_ticker}'] = np.log(
            aligned_data[f'Close_{hedge_ticker}']
            / aligned_data[f'Close_{hedge_ticker}'].shift(1)
        )

        # Drop NaN values created by the shift
        aligned_data.dropna(inplace=True)

        # Correlation Analysis
        correlation = aligned_data[f'Log_Returns_{user_ticker}'].corr(
            aligned_data[f'Log_Returns_{hedge_ticker}']
        )
        st.metric(f"Correlation with {hedge_ticker}", f"{correlation:.4f}")

        # Rolling Correlation
        rolling_corr = aligned_data[
            f'Log_Returns_{user_ticker}'].rolling(window=rolling_window).corr(
            aligned_data[f'Log_Returns_{hedge_ticker}']
        )

        # Plot Log Returns
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(
            aligned_data[f'Log_Returns_{user_ticker}'],
            label=f"{user_ticker} Log Returns",
            color="blue",
            alpha=0.7
        )
        ax1.plot(
            aligned_data[f'Log_Returns_{hedge_ticker}'],
            label=f"{hedge_ticker} Log Returns",
            color=hedge_color,
            alpha=0.7
        )
        ax1.set_title(f"Log Returns: {user_ticker} vs {hedge_ticker}",
                      fontsize=16)
        ax1.set_xlabel("Timestamp", fontsize=14)
        ax1.set_ylabel("Log Returns", fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig1)

        # Plot Rolling Correlation
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(
            rolling_corr,
            label=f"Rolling Correlation ({rolling_window} mins)",
            color='purple'
            )
        ax2.axhline(0, color='red', linestyle='--',
                    linewidth=0.8, label='Zero Line')
        ax2.set_title(f"Rolling Correlation: {user_ticker} vs {hedge_ticker}",
                      fontsize=16)
        ax2.set_xlabel("Timestamp", fontsize=14)
        ax2.set_ylabel("Correlation", fontsize=14)
        ax2.legend()
        ax2.grid(alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig2)


class GarchAnalysis(BaseAnalysis):
    def __init__(self):
        super().__init__()

    def preprocess_log_returns(self, log_ret, scale_threshold=1e-3):
        """
        Preprocess log returns to check and apply scaling if necessary.
        """
        scale_factor = 1  # Default to no scaling

        if abs(log_ret.mean()) < scale_threshold:
            scale_factor = 1000
            log_ret = log_ret * scale_factor
            st.info(
                f"Log returns scaled by a factor of {scale_factor} "
                f"for better optimization."
            )
        else:
            st.info("Log returns are well-scaled. No scaling applied.")

        return log_ret, scale_factor

    def find_best_garch_params(self, log_ret):
        """
        Determine the best p and q parameters for
        a GARCH model based on the lowest BIC.
        """
        p_max, q_max = 3, 3
        best_p, best_q = 0, 0
        lowest_bic = np.inf

        progress_bar = st.progress(0)
        total_iterations = p_max * q_max
        current_iteration = 0

        for p in range(1, p_max + 1):
            for q in range(1, q_max + 1):
                try:
                    model = arch_model(log_ret,
                                       vol='Garch',
                                       p=p,
                                       q=q,
                                       dist='normal')
                    fit = model.fit(disp='off')
                    if fit.bic < lowest_bic:
                        lowest_bic = fit.bic
                        best_p, best_q = p, q
                except Exception as e:
                    st.warning(
                        f"Error fitting model with p={p}, q={q}: {e}"
                    )

                current_iteration += 1
                progress_bar.progress(current_iteration / total_iterations)

        st.success(
            f"Best GARCH parameters: p={best_p}, q={best_q} "
            f"with BIC={lowest_bic:.2f}"
        )
        return best_p, best_q

    def compare_models_and_pick_best(self, log_ret, best_p, best_q):
        """
        Fit GARCH and GJR-GARCH models, compare their BICs,
        and pick the best model.
        """
        with st.spinner('Fitting GARCH models...'):
            fit_garch = arch_model(log_ret, vol="Garch",
                                   p=best_p,
                                   q=best_q,
                                   dist="normal").fit(disp="off")
            garch_bic = fit_garch.bic

            residuals = fit_garch.resid
            conditional_volatility = fit_garch.conditional_volatility
            valid_indices = (
                (~np.isnan(residuals)) &
                (~np.isnan(conditional_volatility)) &
                (conditional_volatility > 0)
            )
            standardized_residuals = residuals[
                valid_indices] / conditional_volatility[valid_indices]

            jb_stat, jb_pvalue = jarque_bera(standardized_residuals)

            if jb_pvalue < 0.05:
                fit_gjr_garch = arch_model(log_ret,
                                           vol="Garch",
                                           p=best_p,
                                           q=best_q,
                                           o=1,
                                           dist="t").fit(disp="off")
                gjr_garch_bic = fit_gjr_garch.bic
                best_model = (
                    fit_gjr_garch if gjr_garch_bic < garch_bic else fit_garch
                )
            else:
                best_model = fit_garch

        return best_model

    def plot_garch_volatility(self, fit_model, user_ticker, best_p, best_q):
        """
        Plot the conditional volatility from a fitted GARCH model.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(fit_model.conditional_volatility,
                label='Conditional Volatility', color='blue')
        ax.set_title(
            f'GARCH({best_p}, {best_q}) Volatility for {user_ticker}',
            fontsize=16
        )
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel('Volatility', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig)

    def analyze_volatility(self, data):
        """
        Main method to perform volatility analysis.
        """
        if data.empty:
            st.error("No data available for analysis.")
            return

        # Used help of chatGPT for Calculating the log returns as my code was failing
        # There is a need to again calculate the log returns
        log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()

        # Preprocess log returns
        log_returns, scale_factor = self.preprocess_log_returns(log_returns)

        # Find best GARCH parameters
        best_p, best_q = self.find_best_garch_params(log_returns)

        # Fit the best model
        best_model = self.compare_models_and_pick_best(
            log_returns, best_p, best_q)

        # Display model summary
        st.subheader("Model Summary")

        # Plot volatility
        self.plot_garch_volatility(
            best_model,
            data.name if hasattr(data, 'name') else "Stock",
            best_p,
            best_q
        )
