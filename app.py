import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(
    page_title="Stock Forecast using ARIMA",
    layout="wide"
)

st.title("📈 Stock Price Forecast using ARIMA")
st.write(
    "Download last 5 years of stock data from YFinance "
    "and forecast stock price for June 2027"
)

# -----------------------------
# User input
# -----------------------------
ticker = st.text_input(
    "Enter stock ticker",
    value="AAPL"
)

if st.button("Generate Forecast"):

    try:

        # -----------------------------
        # Download stock data
        # -----------------------------
        with st.spinner("Downloading stock data..."):

            end_date = datetime.today()
            start_date = end_date - pd.DateOffset(years=5)

            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False
            )

        if data.empty:
            st.error(
                "No stock data found. Please check ticker."
            )
            st.stop()

        # -----------------------------
        # Close price selection
        # -----------------------------
        prices = data["Close"]

        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]

        prices = prices.dropna()

        # -----------------------------
        # Display stock data
        # -----------------------------
        st.subheader("Recent Stock Data")

        st.dataframe(
            data.tail()
        )

        # -----------------------------
        # Historical line chart
        # -----------------------------
        st.subheader(
            "Historical Stock Price (5 Years)"
        )

        fig1, ax1 = plt.subplots(
            figsize=(12, 5)
        )

        ax1.plot(
            prices.index,
            prices.values
        )

        ax1.set_title(
            f"{ticker} Historical Price"
        )

        ax1.set_xlabel(
            "Date"
        )

        ax1.set_ylabel(
            "Price"
        )

        ax1.grid()

        st.pyplot(fig1)

        # -----------------------------
        # Monthly average data
        # -----------------------------
        monthly_prices = prices.resample(
            "ME"
        ).mean()

        monthly_prices = monthly_prices.dropna()

        # -----------------------------
        # Fixed ARIMA order
        # -----------------------------
        arima_order = (1, 1, 1)

        st.write(
            f"ARIMA Model Used: {arima_order}"
        )

        # -----------------------------
        # Train ARIMA model
        # -----------------------------
        model = ARIMA(
            monthly_prices,
            order=arima_order
        )

        fitted_model = model.fit()

        # -----------------------------
        # Forecast to June 2027
        # -----------------------------
        target_date = pd.Timestamp(
            "2027-06-30"
        )

        last_date = monthly_prices.index[-1]

        months_needed = (
            (target_date.year - last_date.year)
            * 12
            +
            (
                target_date.month
                - last_date.month
            )
        )

        if months_needed <= 0:
            months_needed = 1

        forecast = fitted_model.forecast(
            steps=months_needed
        )

        future_dates = pd.date_range(
            start=last_date
            + pd.offsets.MonthEnd(1),
            periods=months_needed,
            freq="ME"
        )

        forecast_df = pd.DataFrame(
            {
                "Date": future_dates,
                "Forecast": forecast
            }
        )

        # -----------------------------
        # June 2027 prediction
        # -----------------------------
        june_forecast = forecast_df[
            forecast_df["Date"]
            .dt.strftime("%Y-%m")
            == "2027-06"
        ]

        if len(june_forecast) > 0:

            predicted_price = (
                june_forecast[
                    "Forecast"
                ].iloc[0]
            )

            st.success(
                f"Predicted Price for June 2027: "
                f"${predicted_price:.2f}"
            )

        # -----------------------------
        # Forecast chart
        # -----------------------------
        st.subheader(
            "Forecast Chart"
        )

        fig2, ax2 = plt.subplots(
            figsize=(12,5)
        )

        ax2.plot(
            monthly_prices.index,
            monthly_prices.values,
            label="Historical"
        )

        ax2.plot(
            forecast_df["Date"],
            forecast_df["Forecast"],
            label="Forecast"
        )

        ax2.axvline(
            pd.Timestamp(
                "2027-06-30"
            ),
            linestyle="--"
        )

        ax2.set_title(
            f"{ticker} Forecast till June 2027"
        )

        ax2.set_xlabel(
            "Date"
        )

        ax2.set_ylabel(
            "Price"
        )

        ax2.legend()

        ax2.grid()

        st.pyplot(fig2)

        # -----------------------------
        # Forecast table
        # -----------------------------
        st.subheader(
            "Forecast Values"
        )

        st.dataframe(
            forecast_df
        )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )
