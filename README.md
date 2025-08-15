TIME SERIES ANALYSIS AND FORECASTING FOR STOCK MARKET

A complete, end-to-end Python project for loading OHLCV stock data, running exploratory analysis, testing stationarity, decomposing seasonality, training multiple forecasting models (ARIMA, SARIMA, Prophet, LSTM), evaluating and plotting predictions, generating a text report, and producing short-term forecasts. It also includes optional utilities for ARIMA auto-tuning, simple rule-based trading signals, portfolio backtesting, and a technical-analysis dashboard.

🔧 Features

Data pipeline: Excel loader → preprocessing (sorting, NaNs) → EDA plots.

Tests & transforms: ADF stationarity test; differencing if needed; seasonal decomposition.

Models:

ARIMA

SARIMA

Prophet (daily + yearly seasonality)

LSTM (sequence model with lookback window)

Evaluation: MSE, RMSE, MAE, MAPE, R²; side-by-side prediction plots.

Reporting: Auto-generated text report (stock_analysis_report.txt).

Forecasting: 30-day forward forecasts (ARIMA & Prophet).

Advanced tools (optional):

optimize_arima_parameters (AIC grid search).

Simple trading signals (BUY/SELL/HOLD) and portfolio simulation.

Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands) + a 4-panel TA dashboard.

Project scaffolding: create_project_structure() creates folders and a starter README.

🧱 Project Structure

create_project_structure() produces:

data/        # Raw/processed files
models/      # Saved models (if you persist them)
results/     # Predictions/plots (you can save here)
notebooks/   # Jupyter notebooks
scripts/     # Standalone scripts
reports/     # Generated reports (e.g., stock_analysis_report.txt)
README.md


You can call this function once to scaffold the repo.

📦 Requirements

Install with pip (Python 3.9+ recommended):

pip install pandas numpy matplotlib seaborn plotly statsmodels prophet scikit-learn tensorflow openpyxl xlrd


Notes:

Prophet package name is prophet (not fbprophet).

warnings is from Python’s standard library—no install needed.

TensorFlow can be heavy; CPU is fine for this demo.

📁 Data Format (Excel)

Minimum required columns:

Date (any parseable date format)

Close

Optional but supported:

Open, High, Low, Volume

The loader expects an Excel file (any sheet name/index). CSV support isn’t included by default.

🚀 Quick Start (Command Line)

Clone and create the structure (optional):

python your_script.py  # the script calls print_requirements(), then main()


Choose a mode when prompted:

1 – Generate sample data and run the full pipeline automatically.

2 – Provide your own Excel file path, sheet name (or index), and price column (default: Close).

That’s it. The script:

Prints step-by-step progress

Opens matplotlib charts

Saves a text report: stock_analysis_report.txt

🧭 Step-by-Step Pipeline

The main runner run_complete_analysis() executes these stages:

Load Excel
load_excel_data(file_path, sheet_name)
Reads Excel, prints shape/columns + head preview.

Preprocess
preprocess_data(date_column='Date', price_column='Close')

Parse dates → set index

Sort by date

Forward/backward fill missing values

Print stats and date range

EDA
exploratory_data_analysis(price_column)

Close price trend

Volume trend (if available)

Close price histogram

Daily returns histogram + summary stats

Stationarity (ADF)
check_stationarity(price_column)
Prints ADF statistic, p-value, critical values; returns True/False.

Seasonal Decomposition
seasonal_decomposition(price_column, period=252)
Plots observed/trend/seasonal/residual.

(If needed) Differencing
make_stationary(price_column)
First difference, ADF; if still non-stationary, second difference.

ACF/PACF
plot_acf_pacf(data, lags=40)
Plots ACF/PACF for ARIMA order intuition.

Model Training

ARIMA: train_arima_model(order=(1,1,1))

SARIMA: train_sarima_model(order=(1,1,1), seasonal_order=(1,1,1,12))

Prophet: train_prophet_model() (uses ds, y format)

LSTM:

prepare_lstm_data(lookback=60) scales Close + builds sequences

train_lstm_model(lookback=60, epochs=50)

All models use an 80/20 split (chronological).

Evaluation
evaluate_models() computes MSE, RMSE, MAE, MAPE, R² for each model.
Also: plot_predictions() gives a 2×2 comparison figure (ARIMA, SARIMA, Prophet, LSTM).

Reporting
create_comprehensive_report(evaluation_results)
Saves stock_analysis_report.txt with dataset stats, model table, best model, and volatility insights.

Forecasting
forecast_future(days=30)
Produces 30-day forward curves for ARIMA and Prophet and overlays them after the historical window.

⚙️ Configuration & Customization

Price column: pass price_column='Adj Close' (or any) in run_complete_analysis.

ARIMA/SARIMA orders: edit train_arima_model() / train_sarima_model() arguments.

Prophet: seasonalities are enabled (daily, yearly); adjust inside train_prophet_model().

LSTM: tweak lookback, epochs, layers/units, dropout rates.

Decomposition period: default 252 (trading days); change via seasonal_decomposition(period=...).

Forecast horizon: change days in forecast_future(days=...).

📊 Models & What They Do

ARIMA(p,d,q): Captures autoregression, differencing, and moving average on stationary series.

SARIMA(p,d,q)(P,D,Q,s): Adds explicit seasonality (period s, default 12).

Prophet: Trend + seasonality with additive components, robust to missing data/outliers.

LSTM: Sequence model learning temporal dependencies on scaled data with sliding windows.

All models train on the first 80% of observations and validate on the final 20%.

📐 Metrics

For each model (on the test split):

MSE, RMSE, MAE, MAPE, R²

The report highlights the best model by RMSE.

📈 Outputs You’ll See

Multiple matplotlib windows:

EDA panels

Decomposition

ACF/PACF

Model comparison (2×2)

Forecast overlay

Text file:

stock_analysis_report.txt (comprehensive summary)

🧪 Advanced (Optional)

Auto-tune ARIMA (AIC)
best = optimize_arima_parameters(close_series, max_p=3, max_d=2, max_q=3)

Trading signals

signals = create_trading_signals(predictions, actual_prices, threshold=0.02)
values, total_return = calculate_portfolio_performance(signals, actual_prices, initial_capital=10000)


Technical Analysis Dashboard

adv = AdvancedAnalyzer()
adv.data = your_dataframe
adv.preprocess_data(price_column='Close')
adv.calculate_technical_indicators()
adv.plot_technical_analysis()

🧩 Programmatic Usage
from your_script import StockTimeSeriesAnalyzer

analyzer = StockTimeSeriesAnalyzer()
analyzer.run_complete_analysis(
    file_path="data/your_file.xlsx",
    sheet_name=0,           # or "Sheet1"
    price_column="Close"    # or any column name
)


To scaffold folders:

from your_script import create_project_structure
create_project_structure()

🛠️ Troubleshooting

Prophet install issues: make sure you installed prophet and have a compatible Python toolchain.

Excel read errors: verify file path, permissions, and that the sheet exists.

Long training times: reduce LSTM epochs, or skip LSTM if not needed.

Empty/short datasets: ensure Date/Close exist and you have enough rows for splits and lookbacks.

⚠️ Disclaimer

This repository is for educational and research purposes. It is not financial advice. Forecasts are uncertain; always validate and use proper risk management.

📄 License

Choose a license appropriate for your use (e.g., MIT). Add a LICENSE file at the repo root.

🤝 Contributing

Issues and PRs are welcome. Please keep changes consistent with the project’s design:

Clear docstrings

Deterministic splits

Minimal external assumptions

✅ TL;DR

Put your Excel file in data/, or use the built-in sample generator.

Run the script → follow the prompt → get plots and stock_analysis_report.txt.

Tune models as needed; optional tools help with auto-tuning, signals, and TA plots.

