import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

st.set_page_config(
    page_title="Stock Forecast using ARIMA",
    layout="wide"
)

st.title("📈 Stock Forecast using ARIMA")
st.write(
    "Downloads last 5 years of stock data from YFinance "
    "and forecasts price for June 2027."
)

ticker = st.text_input(
    "Enter stock ticker:",
    value="AAPL"
)

if st.button("Generate Forecast"):

    try:
        with st.spinner("Downloading stock data..."):

            end_date = datetime.today()
            start_date = end_date - pd.DateOffset(years=5)

            stock_data = yf.download(
                ticker,
                start=start_date,
                end=end_date
            )

        if stock_data.empty:
            st.error("No data found. Check ticker symbol.")
            st.stop()

        prices = stock_data["Close"]

        st.subheader("Historical Data")
        st.dataframe(stock_data.tail())

        # Historical chart
        st.subheader("Last 5 Years Price Chart")

        fig1, ax1 = plt.subplots(figsize=(12,5))

        ax1.plot(
            prices.index,
            prices.values
        )

        ax1.set_title(
            f"{ticker} Historical Price"
        )

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")

        ax1.grid()

        st.pyplot(fig1)

        # Monthly resample
        monthly_prices = prices.resample("M").mean()

        st.write("Training Auto-ARIMA model...")

        auto_model = auto_arima(
            monthly_prices,
            seasonal=False,
            suppress_warnings=True,
            error_action="ignore"
        )

        best_order = auto_model.order

        st.write(
            f"Selected ARIMA Order: {best_order}"
        )

        model = ARIMA(
            monthly_prices,
            order=best_order
        )

        fitted_model = model.fit()

        future_target = pd.Timestamp(
            "2027-06-30"
        )

        months_needed = (
            (future_target.year -
             monthly_prices.index[-1].year) * 12
            +
            (future_target.month -
             monthly_prices.index[-1].month)
        )

        forecast = fitted_model.forecast(
            steps=months_needed
        )

        future_dates = pd.date_range(
            start=monthly_prices.index[-1]
                  + pd.offsets.MonthEnd(),
            periods=months_needed,
            freq="M"
        )

        forecast_df = pd.DataFrame(
            {
                "Date": future_dates,
                "Forecast": forecast
            }
        )

        june_2027 = forecast_df[
            forecast_df["Date"]
            .dt.strftime("%Y-%m")
            == "2027-06"
        ]

        predicted_price = float(
            june_2027["Forecast"].iloc[0]
        )

        st.success(
            f"Predicted price for June 2027: "
            f"{predicted_price:.2f}"
        )

        st.subheader("Forecast Chart")

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
            pd.Timestamp("2027-06-30"),
            linestyle="--"
        )

        ax2.legend()

        ax2.set_title(
            f"{ticker} Forecast until June 2027"
        )

        ax2.grid()

        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error: {e}")
