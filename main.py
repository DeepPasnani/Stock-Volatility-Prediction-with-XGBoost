import sys
import pandas as pd
from ml import run_prediction


def run_streamlit():
    import streamlit as st

    st.title("Stock Volatility Prediction App")
    ticker = st.text_input("Ticker Symbol (e.g. AAPL)", "AAPL")
    start = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
    end = st.date_input("End Date", pd.to_datetime("2025-08-13"))

    if st.button("Predict Volatility"):
        with st.spinner("Downloading and processing data..."):
            try:
                result = run_prediction(
                    ticker,
                    start.strftime("%Y-%m-%d"),
                    end.strftime("%Y-%m-%d"),
                )
            except ValueError as e:
                st.error(str(e))
                return

        preds = result["predictions"]
        chart_df = pd.DataFrame(
            {"Actual": [p["actual"] for p in preds], "Predicted": [p["predicted"] for p in preds]},
            index=[p["date"] for p in preds],
        )
        st.line_chart(chart_df)
        st.write(f"RMSE: {result['rmse']:.4f}")
        st.write(f"R²: {result['r2']:.4f}")

        st.subheader("Feature Importance")
        feat_df = pd.DataFrame(result["feature_importance"])
        if not feat_df.empty:
            feat_df = feat_df.set_index("feature")
            st.bar_chart(feat_df)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        run_streamlit()
    else:
        print("Enter stock ticker, start date (YYYY-MM-DD), and end date (YYYY-MM-DD):")
        ticker = input("Ticker: ").strip().upper()
        start = input("Start Date (YYYY-MM-DD): ").strip()
        end = input("End Date (YYYY-MM-DD): ").strip()
        try:
            result = run_prediction(ticker, start, end)
            print(f"Ticker: {result['ticker']}")
            print(f"RMSE: {result['rmse']:.4f}")
            print(f"R²: {result['r2']:.4f}")
            print(f"Train rows: {result['train_rows']}")
            print(f"Test rows: {result['test_rows']}")
            print(f"Total rows: {result['total_rows']}")
        except ValueError as e:
            print(f"Error: {e}")
