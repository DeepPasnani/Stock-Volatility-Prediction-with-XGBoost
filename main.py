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
    elif len(sys.argv) > 1 and sys.argv[1] == "tune":
        from ml.tune import tune_hyperparameters

        if len(sys.argv) < 5:
            print("Usage: python main.py tune TICKER START_DATE END_DATE [--folds N]")
            sys.exit(1)

        ticker = sys.argv[2].upper()
        start = sys.argv[3]
        end = sys.argv[4]
        folds = 3
        if "--folds" in sys.argv:
            idx = sys.argv.index("--folds")
            if idx + 1 < len(sys.argv):
                folds = int(sys.argv[idx + 1])

        try:
            result = tune_hyperparameters(ticker, start, end, cv_folds=folds)
            print(f"\nTuning results for {ticker}:")
            print(f"  Best CV score (RMSE): {result['best_score']:.6f}")
            print(f"  Best params: {result['best_params']}")
            print(f"  CV folds: {result['cv_folds']}")
            print(f"  Total rows: {result['total_rows']}")
            print(f"  Combinations tested: {len(result['cv_results'])}")
            print("\nAll results:")
            for r in result["cv_results"]:
                print(f"    {r['params']}  RMSE={r['mean_test_score']:.6f} ±{r['std_test_score']:.6f}")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("Usage:")
        print("  python main.py                    # CLI mode (interactive)")
        print("  python main.py streamlit          # Streamlit UI")
        print("  python main.py tune TICKER START END [--folds N]  # Hyperparameter tuning")
        sys.exit(1)
