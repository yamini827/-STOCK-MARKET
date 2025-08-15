
**TIME SERIES ANALYSIS AND FORECASTING FOR STOCK MARKET**
"""

# Stock Market Time Series Analysis and Forecasting Project
# Complete implementation with step-by-step process

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet

# Deep learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# File handling
import openpyxl
from datetime import datetime, timedelta
import os

class StockTimeSeriesAnalyzer:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.models = {}
        self.predictions = {}
        self.scaler = MinMaxScaler()

    def load_excel_data(self, file_path, sheet_name=0):
        """
        Load stock data from Excel file
        Expected columns: Date, Open, High, Low, Close, Volume
        """
        print("📊 Step 1: Loading Excel Data...")

        try:
            # Load data from Excel
            self.data = pd.read_excel(file_path, sheet_name=sheet_name)
            print(f"✅ Data loaded successfully!")
            print(f"📈 Dataset shape: {self.data.shape}")
            print(f"📅 Columns: {list(self.data.columns)}")

            # Display first few rows
            print("\n🔍 First 5 rows of data:")
            print(self.data.head())

            return True

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def preprocess_data(self, date_column='Date', price_column='Close'):
        """
        Preprocess the stock data for time series analysis
        """
        print("\n🔧 Step 2: Data Preprocessing...")

        # Make a copy of original data
        self.processed_data = self.data.copy()

        # Convert date column to datetime
        if date_column in self.processed_data.columns:
            self.processed_data[date_column] = pd.to_datetime(self.processed_data[date_column])
            self.processed_data.set_index(date_column, inplace=True)

        # Sort by date
        self.processed_data.sort_index(inplace=True)

        # Handle missing values
        print(f"📋 Missing values before cleaning: {self.processed_data.isnull().sum().sum()}")
        self.processed_data.fillna(method='ffill', inplace=True)  # Forward fill
        self.processed_data.fillna(method='bfill', inplace=True)  # Backward fill

        # Remove any remaining missing values
        self.processed_data.dropna(inplace=True)

        print(f"✅ Missing values after cleaning: {self.processed_data.isnull().sum().sum()}")
        print(f"📊 Final dataset shape: {self.processed_data.shape}")
        print(f"📅 Date range: {self.processed_data.index.min()} to {self.processed_data.index.max()}")

        # Basic statistics
        print("\n📈 Basic Statistics for Close Price:")
        print(self.processed_data[price_column].describe())

    def exploratory_data_analysis(self, price_column='Close'):
        """
        Perform comprehensive EDA on the stock data
        """
        print("\n📊 Step 3: Exploratory Data Analysis...")

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Stock Market Data - Exploratory Data Analysis', fontsize=16)

        # 1. Price trend over time
        axes[0, 0].plot(self.processed_data.index, self.processed_data[price_column])
        axes[0, 0].set_title(f'{price_column} Price Trend')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Volume trend
        if 'Volume' in self.processed_data.columns:
            axes[0, 1].plot(self.processed_data.index, self.processed_data['Volume'], color='orange')
            axes[0, 1].set_title('Volume Trend')
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel('Volume')
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Price distribution
        axes[1, 0].hist(self.processed_data[price_column], bins=50, alpha=0.7, color='green')
        axes[1, 0].set_title(f'{price_column} Price Distribution')
        axes[1, 0].set_xlabel('Price')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Daily returns
        daily_returns = self.processed_data[price_column].pct_change().dropna()
        axes[1, 1].hist(daily_returns, bins=50, alpha=0.7, color='red')
        axes[1, 1].set_title('Daily Returns Distribution')
        axes[1, 1].set_xlabel('Daily Return')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Calculate and display daily returns statistics
        print(f"\n📈 Daily Returns Statistics:")
        print(f"Mean: {daily_returns.mean():.4f}")
        print(f"Std: {daily_returns.std():.4f}")
        print(f"Skewness: {daily_returns.skew():.4f}")
        print(f"Kurtosis: {daily_returns.kurtosis():.4f}")

    def check_stationarity(self, price_column='Close'):
        """
        Check stationarity using Augmented Dickey-Fuller test
        """
        print("\n🔍 Step 4: Stationarity Analysis...")

        # Perform ADF test
        result = adfuller(self.processed_data[price_column])

        print(f"📊 Augmented Dickey-Fuller Test Results:")
        print(f"ADF Statistic: {result[0]:.6f}")
        print(f"p-value: {result[1]:.6f}")
        print(f"Critical Values:")
        for key, value in result[4].items():
            print(f"\t{key}: {value:.3f}")

        if result[1] <= 0.05:
            print("✅ Series is stationary (reject null hypothesis)")
            return True
        else:
            print("❌ Series is non-stationary (fail to reject null hypothesis)")
            return False

    def seasonal_decomposition(self, price_column='Close', period=252):
        """
        Perform seasonal decomposition
        """
        print("\n📈 Step 5: Seasonal Decomposition...")

        # Perform decomposition
        decomposition = seasonal_decompose(self.processed_data[price_column],
                                         model='additive',
                                         period=period)

        # Plot decomposition
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle('Time Series Decomposition', fontsize=16)

        decomposition.observed.plot(ax=axes[0], title='Original')
        decomposition.trend.plot(ax=axes[1], title='Trend')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
        decomposition.resid.plot(ax=axes[3], title='Residual')

        plt.tight_layout()
        plt.show()

        return decomposition

    def make_stationary(self, price_column='Close'):
        """
        Make the time series stationary using differencing
        """
        print("\n🔄 Making series stationary...")

        # First difference
        diff_data = self.processed_data[price_column].diff().dropna()

        # Check stationarity of differenced data
        result = adfuller(diff_data)
        print(f"📊 ADF test on differenced data:")
        print(f"p-value: {result[1]:.6f}")

        if result[1] <= 0.05:
            print("✅ First difference made the series stationary")
            return diff_data, 1
        else:
            # Second difference
            diff2_data = diff_data.diff().dropna()
            result2 = adfuller(diff2_data)
            print(f"📊 ADF test on second differenced data:")
            print(f"p-value: {result2[1]:.6f}")

            if result2[1] <= 0.05:
                print("✅ Second difference made the series stationary")
                return diff2_data, 2
            else:
                print("⚠️ Series still not stationary after second differencing")
                return diff2_data, 2

    def plot_acf_pacf(self, data, lags=40):
        """
        Plot ACF and PACF for ARIMA model order selection
        """
        print("\n📊 Step 6: ACF and PACF Analysis...")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('ACF and PACF Plots', fontsize=16)

        plot_acf(data, ax=axes[0], lags=lags)
        plot_pacf(data, ax=axes[1], lags=lags)

        plt.tight_layout()
        plt.show()

    def train_arima_model(self, price_column='Close', order=(1,1,1)):
        """
        Train ARIMA model
        """
        print(f"\n🤖 Step 7a: Training ARIMA Model with order {order}...")

        try:
            # Split data
            train_size = int(len(self.processed_data) * 0.8)
            train_data = self.processed_data[price_column][:train_size]
            test_data = self.processed_data[price_column][train_size:]

            # Train ARIMA model
            model = ARIMA(train_data, order=order)
            fitted_model = model.fit()

            print("✅ ARIMA model trained successfully!")
            print(fitted_model.summary())

            # Make predictions
            predictions = fitted_model.forecast(steps=len(test_data))

            # Store model and predictions
            self.models['ARIMA'] = fitted_model
            self.predictions['ARIMA'] = {
                'train': train_data,
                'test': test_data,
                'predictions': predictions,
                'train_size': train_size
            }

            return fitted_model

        except Exception as e:
            print(f"❌ Error training ARIMA model: {e}")
            return None

    def train_sarima_model(self, price_column='Close', order=(1,1,1), seasonal_order=(1,1,1,12)):
        """
        Train SARIMA model
        """
        print(f"\n🤖 Step 7b: Training SARIMA Model...")
        print(f"Order: {order}, Seasonal Order: {seasonal_order}")

        try:
            # Split data
            train_size = int(len(self.processed_data) * 0.8)
            train_data = self.processed_data[price_column][:train_size]
            test_data = self.processed_data[price_column][train_size:]

            # Train SARIMA model
            model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit()

            print("✅ SARIMA model trained successfully!")

            # Make predictions
            predictions = fitted_model.forecast(steps=len(test_data))

            # Store model and predictions
            self.models['SARIMA'] = fitted_model
            self.predictions['SARIMA'] = {
                'train': train_data,
                'test': test_data,
                'predictions': predictions,
                'train_size': train_size
            }

            return fitted_model

        except Exception as e:
            print(f"❌ Error training SARIMA model: {e}")
            return None

    def train_prophet_model(self, price_column='Close'):
        """
        Train Facebook Prophet model
        """
        print("\n🤖 Step 7c: Training Prophet Model...")

        try:
            # Prepare data for Prophet
            prophet_data = self.processed_data.reset_index()
            prophet_data = prophet_data[['Date', price_column]].rename(
                columns={'Date': 'ds', price_column: 'y'}
            )

            # Split data
            train_size = int(len(prophet_data) * 0.8)
            train_data = prophet_data[:train_size]
            test_data = prophet_data[train_size:]

            # Train Prophet model
            model = Prophet(daily_seasonality=True, yearly_seasonality=True)
            model.fit(train_data)

            print("✅ Prophet model trained successfully!")

            # Make predictions
            future = model.make_future_dataframe(periods=len(test_data))
            forecast = model.predict(future)

            # Store model and predictions
            self.models['Prophet'] = model
            self.predictions['Prophet'] = {
                'train': train_data,
                'test': test_data,
                'forecast': forecast,
                'train_size': train_size
            }

            return model

        except Exception as e:
            print(f"❌ Error training Prophet model: {e}")
            return None

    def prepare_lstm_data(self, price_column='Close', lookback=60):
        """
        Prepare data for LSTM model
        """
        # Scale data
        scaled_data = self.scaler.fit_transform(self.processed_data[[price_column]])

        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        return X, y, scaled_data

    def train_lstm_model(self, price_column='Close', lookback=60, epochs=50):
        """
        Train LSTM model
        """
        print(f"\n🤖 Step 7d: Training LSTM Model...")
        print(f"Lookback window: {lookback}, Epochs: {epochs}")

        try:
            # Prepare data
            X, y, scaled_data = self.prepare_lstm_data(price_column, lookback)

            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])

            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

            print("🏋️ Training LSTM model...")
            history = model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=epochs,
                validation_data=(X_test, y_test),
                verbose=0
            )

            print("✅ LSTM model trained successfully!")

            # Make predictions
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)

            # Inverse transform predictions
            train_predictions = self.scaler.inverse_transform(train_predictions)
            test_predictions = self.scaler.inverse_transform(test_predictions)

            # Store model and predictions
            self.models['LSTM'] = model
            self.predictions['LSTM'] = {
                'train_predictions': train_predictions.flatten(),
                'test_predictions': test_predictions.flatten(),
                'train_size': train_size,
                'lookback': lookback,
                'history': history,
                'y_train': self.scaler.inverse_transform(y_train.reshape(-1, 1)).flatten(),
                'y_test': self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            }

            return model

        except Exception as e:
            print(f"❌ Error training LSTM model: {e}")
            return None

    def calculate_metrics(self, actual, predicted):
        """
        Calculate evaluation metrics
        """
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)

        # Calculate MAPE (avoiding division by zero)
        mape = np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, 1))) * 100

        # Calculate R²
        r2 = r2_score(actual, predicted)

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R²': r2
        }

    def evaluate_models(self, price_column='Close'):
        """
        Evaluate and compare all trained models
        """
        print("\n📊 Step 8: Model Evaluation and Comparison...")

        results = {}

        # Evaluate ARIMA
        if 'ARIMA' in self.predictions:
            pred_data = self.predictions['ARIMA']
            metrics = self.calculate_metrics(pred_data['test'], pred_data['predictions'])
            results['ARIMA'] = metrics
            print(f"✅ ARIMA Metrics: RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, MAPE={metrics['MAPE']:.2f}%")

        # Evaluate SARIMA
        if 'SARIMA' in self.predictions:
            pred_data = self.predictions['SARIMA']
            metrics = self.calculate_metrics(pred_data['test'], pred_data['predictions'])
            results['SARIMA'] = metrics
            print(f"✅ SARIMA Metrics: RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, MAPE={metrics['MAPE']:.2f}%")

        # Evaluate Prophet
        if 'Prophet' in self.predictions:
            pred_data = self.predictions['Prophet']
            test_forecast = pred_data['forecast']['yhat'][pred_data['train_size']:].values
            metrics = self.calculate_metrics(pred_data['test']['y'], test_forecast)
            results['Prophet'] = metrics
            print(f"✅ Prophet Metrics: RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, MAPE={metrics['MAPE']:.2f}%")

        # Evaluate LSTM
        if 'LSTM' in self.predictions:
            pred_data = self.predictions['LSTM']
            metrics = self.calculate_metrics(pred_data['y_test'], pred_data['test_predictions'])
            results['LSTM'] = metrics
            print(f"✅ LSTM Metrics: RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, MAPE={metrics['MAPE']:.2f}%")

        return results

    def plot_predictions(self, price_column='Close'):
        """
        Plot predictions from all models
        """
        print("\n📈 Step 9: Plotting Predictions...")

        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Model Predictions Comparison', fontsize=16)

        # ARIMA
        if 'ARIMA' in self.predictions:
            pred_data = self.predictions['ARIMA']
            ax = axes[0, 0]
            ax.plot(pred_data['train'].index, pred_data['train'], label='Training Data', color='blue')
            ax.plot(pred_data['test'].index, pred_data['test'], label='Actual', color='green')
            ax.plot(pred_data['test'].index, pred_data['predictions'], label='ARIMA Predictions', color='red')
            ax.set_title('ARIMA Model')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # SARIMA
        if 'SARIMA' in self.predictions:
            pred_data = self.predictions['SARIMA']
            ax = axes[0, 1]
            ax.plot(pred_data['train'].index, pred_data['train'], label='Training Data', color='blue')
            ax.plot(pred_data['test'].index, pred_data['test'], label='Actual', color='green')
            ax.plot(pred_data['test'].index, pred_data['predictions'], label='SARIMA Predictions', color='red')
            ax.set_title('SARIMA Model')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Prophet
        if 'Prophet' in self.predictions:
            pred_data = self.predictions['Prophet']
            ax = axes[1, 0]
            ax.plot(pred_data['train']['ds'], pred_data['train']['y'], label='Training Data', color='blue')
            ax.plot(pred_data['test']['ds'], pred_data['test']['y'], label='Actual', color='green')
            test_forecast = pred_data['forecast']['yhat'][pred_data['train_size']:].values
            ax.plot(pred_data['test']['ds'], test_forecast, label='Prophet Predictions', color='red')
            ax.set_title('Prophet Model')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # LSTM
        if 'LSTM' in self.predictions:
            pred_data = self.predictions['LSTM']
            ax = axes[1, 1]

            # Create indices for plotting
            train_idx = range(pred_data['lookback'], pred_data['lookback'] + len(pred_data['y_train']))
            test_idx = range(pred_data['lookback'] + len(pred_data['y_train']),
                           pred_data['lookback'] + len(pred_data['y_train']) + len(pred_data['y_test']))

            ax.plot(train_idx, pred_data['y_train'], label='Training Data', color='blue')
            ax.plot(test_idx, pred_data['y_test'], label='Actual', color='green')
            ax.plot(test_idx, pred_data['test_predictions'], label='LSTM Predictions', color='red')
            ax.set_title('LSTM Model')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def create_comprehensive_report(self, evaluation_results, price_column='Close'):
        """
        Create a comprehensive analysis report
        """
        print("\n📋 Step 10: Creating Comprehensive Report...")

        report = []
        report.append("=" * 80)
        report.append("STOCK MARKET TIME SERIES ANALYSIS - COMPREHENSIVE REPORT")
        report.append("=" * 80)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Dataset Shape: {self.processed_data.shape}")
        report.append(f"Date Range: {self.processed_data.index.min()} to {self.processed_data.index.max()}")
        report.append(f"Target Variable: {price_column}")
        report.append("")

        # Dataset Statistics
        report.append("DATASET STATISTICS")
        report.append("-" * 40)
        stats = self.processed_data[price_column].describe()
        for stat, value in stats.items():
            report.append(f"{stat.capitalize()}: {value:.2f}")
        report.append("")

        # Model Performance Comparison
        report.append("MODEL PERFORMANCE COMPARISON")
        report.append("-" * 40)
        report.append(f"{'Model':<10} {'RMSE':<10} {'MAE':<10} {'MAPE':<10} {'R²':<10}")
        report.append("-" * 50)

        for model_name, metrics in evaluation_results.items():
            report.append(f"{model_name:<10} {metrics['RMSE']:<10.2f} {metrics['MAE']:<10.2f} {metrics['MAPE']:<10.2f}% {metrics['R²']:<10.3f}")

        report.append("")

        # Best Model
        if evaluation_results:
            best_model = min(evaluation_results.keys(), key=lambda x: evaluation_results[x]['RMSE'])
            report.append("BEST PERFORMING MODEL")
            report.append("-" * 40)
            report.append(f"Model: {best_model}")
            report.append(f"RMSE: {evaluation_results[best_model]['RMSE']:.2f}")
            report.append(f"MAE: {evaluation_results[best_model]['MAE']:.2f}")
            report.append(f"MAPE: {evaluation_results[best_model]['MAPE']:.2f}%")
            report.append(f"R²: {evaluation_results[best_model]['R²']:.3f}")

        report.append("")
        report.append("ANALYSIS INSIGHTS")
        report.append("-" * 40)

        # Generate insights based on data
        daily_returns = self.processed_data[price_column].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility

        report.append(f"• Average Daily Return: {daily_returns.mean():.4f}")
        report.append(f"• Daily Volatility: {daily_returns.std():.4f}")
        report.append(f"• Annualized Volatility: {volatility:.4f}")
        report.append(f"• Maximum Daily Gain: {daily_returns.max():.4f}")
        report.append(f"• Maximum Daily Loss: {daily_returns.min():.4f}")

        report.append("")
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        report.append("• Use ensemble methods combining multiple models for better accuracy")
        report.append("• Consider external factors (news, economic indicators) for improved predictions")
        report.append("• Regular model retraining is recommended for changing market conditions")
        report.append("• Risk management strategies should account for high volatility periods")

        report.append("")
        report.append("=" * 80)

        # Print report
        for line in report:
            print(line)

        # Save report to file
        with open('stock_analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))
        print("\n💾 Report saved to 'stock_analysis_report.txt'")

        return report

    def forecast_future(self, days=30, price_column='Close'):
        """
        Generate future forecasts using trained models
        """
        print(f"\n🔮 Step 11: Generating {days}-day Future Forecasts...")

        future_forecasts = {}

        # ARIMA Future Forecast
        if 'ARIMA' in self.models:
            try:
                arima_forecast = self.models['ARIMA'].forecast(steps=days)
                future_forecasts['ARIMA'] = arima_forecast
                print(f"✅ ARIMA forecast generated")
            except Exception as e:
                print(f"❌ ARIMA forecast error: {e}")

        # Prophet Future Forecast
        if 'Prophet' in self.models:
            try:
                future = self.models['Prophet'].make_future_dataframe(periods=days)
                prophet_forecast = self.models['Prophet'].predict(future)
                future_forecasts['Prophet'] = prophet_forecast['yhat'][-days:].values
                print(f"✅ Prophet forecast generated")
            except Exception as e:
                print(f"❌ Prophet forecast error: {e}")

        # Plot future forecasts
        if future_forecasts:
            plt.figure(figsize=(15, 8))

            # Plot historical data
            plt.plot(self.processed_data.index[-100:],
                    self.processed_data[price_column][-100:],
                    label='Historical Data', color='blue')

            # Create future dates
            last_date = self.processed_data.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')

            # Plot forecasts
            colors = ['red', 'green', 'orange', 'purple']
            for i, (model_name, forecast) in enumerate(future_forecasts.items()):
                plt.plot(future_dates, forecast,
                        label=f'{model_name} Forecast',
                        color=colors[i % len(colors)],
                        linestyle='--')

            plt.title(f'{days}-Day Future Price Forecasts')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        return future_forecasts

    def run_complete_analysis(self, file_path, sheet_name=0, price_column='Close'):
        """
        Run the complete time series analysis pipeline
        """
        print("🚀 STARTING COMPLETE STOCK MARKET TIME SERIES ANALYSIS")
        print("=" * 80)

        try:
            # Step 1: Load data
            if not self.load_excel_data(file_path, sheet_name):
                return False

            # Step 2: Preprocess data
            self.preprocess_data(price_column=price_column)

            # Step 3: EDA
            self.exploratory_data_analysis(price_column)

            # Step 4: Check stationarity
            is_stationary = self.check_stationarity(price_column)

            # Step 5: Seasonal decomposition
            decomposition = self.seasonal_decomposition(price_column)

            # Step 6: Make stationary if needed
            if not is_stationary:
                stationary_data, diff_order = self.make_stationary(price_column)
                self.plot_acf_pacf(stationary_data)
            else:
                self.plot_acf_pacf(self.processed_data[price_column])

            # Step 7: Train all models
            print("\n🤖 TRAINING ALL MODELS...")

            # ARIMA
            self.train_arima_model(price_column, order=(1,1,1))

            # SARIMA
            self.train_sarima_model(price_column, order=(1,1,1), seasonal_order=(1,1,1,12))

            # Prophet
            self.train_prophet_model(price_column)

            # LSTM
            self.train_lstm_model(price_column, lookback=60, epochs=50)

            # Step 8: Evaluate models
            evaluation_results = self.evaluate_models(price_column)

            # Step 9: Plot predictions
            self.plot_predictions(price_column)

            # Step 10: Create comprehensive report
            self.create_comprehensive_report(evaluation_results, price_column)

            # Step 11: Future forecasting
            future_forecasts = self.forecast_future(days=30, price_column=price_column)

            print("\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 80)

            return True

        except Exception as e:
            print(f"❌ Error in analysis pipeline: {e}")
            return False


# Example usage and sample data creation
def create_sample_data():
    """
    Create sample stock data and save to Excel file
    """
    print("📝 Creating sample stock data...")

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')

    # Generate price data with trend and seasonality
    trend = np.linspace(100, 150, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.normal(0, 5, len(dates))
    close_prices = trend + seasonal + noise

    # Generate other OHLC data
    open_prices = close_prices + np.random.normal(0, 1, len(dates))
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.normal(0, 2, len(dates)))
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.normal(0, 2, len(dates)))
    volumes = np.random.randint(100000, 1000000, len(dates))

    # Create DataFrame
    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    })

    # Save to Excel
    sample_data.to_excel('sample_stock_data.xlsx', index=False)
    print("✅ Sample data saved to 'sample_stock_data.xlsx'")

    return sample_data


# Main execution function
def main():
    """
    Main function to demonstrate the complete workflow
    """
    print("🎯 STOCK MARKET TIME SERIES ANALYSIS PROJECT")
    print("=" * 80)

    # Option 1: Create sample data for demonstration
    print("\n📊 Would you like to:")
    print("1. Use sample data (demonstration)")
    print("2. Use your own Excel file")

    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == '1':
        # Create and use sample data
        sample_data = create_sample_data()
        file_path = 'sample_stock_data.xlsx'

        # Initialize analyzer
        analyzer = StockTimeSeriesAnalyzer()

        # Run complete analysis
        success = analyzer.run_complete_analysis(
            file_path=file_path,
            sheet_name=0,
            price_column='Close'
        )

        if success:
            print("✅ Sample analysis completed successfully!")

    elif choice == '2':
        # Use user's own data
        file_path = input("Enter the path to your Excel file: ").strip()
        sheet_name = input("Enter sheet name or index (press Enter for default): ").strip()
        if not sheet_name:
            sheet_name = 0
        else:
            try:
                sheet_name = int(sheet_name)
            except ValueError:
                pass  # Keep as string for sheet name

        price_column = input("Enter the price column name (default: Close): ").strip()
        if not price_column:
            price_column = 'Close'

        # Initialize analyzer
        analyzer = StockTimeSeriesAnalyzer()

        # Run complete analysis
        success = analyzer.run_complete_analysis(
            file_path=file_path,
            sheet_name=sheet_name,
            price_column=price_column
        )

        if success:
            print("✅ Analysis completed successfully!")
        else:
            print("❌ Analysis failed. Please check your file and parameters.")

    else:
        print("❌ Invalid choice. Please run the script again.")


# Installation requirements
def print_requirements():
    """
    Print required packages for installation
    """
    requirements = """
    📦 REQUIRED PACKAGES:

    pip install pandas numpy matplotlib seaborn plotly
    pip install statsmodels prophet scikit-learn
    pip install tensorflow openpyxl xlrd
    pip install warnings

    🔧 OR INSTALL ALL AT ONCE:
    pip install pandas numpy matplotlib seaborn plotly statsmodels prophet scikit-learn tensorflow openpyxl xlrd

    📋 EXPECTED EXCEL FILE FORMAT:
    - Column 1: Date (any date format)
    - Column 2: Open (opening price)
    - Column 3: High (highest price)
    - Column 4: Low (lowest price)
    - Column 5: Close (closing price)
    - Column 6: Volume (trading volume)

    Note: At minimum, you need Date and Close columns.
    """
    print(requirements)


if __name__ == "__main__":
    print_requirements()
    print("\n" + "="*80)
    main()


# ===============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# ===============================================================================

def optimize_arima_parameters(data, max_p=3, max_d=2, max_q=3):
    """
    Find optimal ARIMA parameters using AIC criterion
    """
    print("🔍 Optimizing ARIMA parameters...")

    best_aic = np.inf
    best_params = None

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(data, order=(p, d, q))
                    fitted_model = model.fit()
                    aic = fitted_model.aic

                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)

                except:
                    continue

    print(f"✅ Best ARIMA parameters: {best_params} with AIC: {best_aic:.2f}")
    return best_params


def create_trading_signals(predictions, actual_prices, threshold=0.02):
    """
    Generate trading signals based on predictions
    """
    signals = []

    for i in range(len(predictions)):
        if i == 0:
            signals.append('HOLD')
            continue

        price_change = (predictions[i] - actual_prices[i-1]) / actual_prices[i-1]

        if price_change > threshold:
            signals.append('BUY')
        elif price_change < -threshold:
            signals.append('SELL')
        else:
            signals.append('HOLD')

    return signals


def calculate_portfolio_performance(signals, prices, initial_capital=10000):
    """
    Calculate portfolio performance based on trading signals
    """
    capital = initial_capital
    shares = 0
    portfolio_values = []

    for i, (signal, price) in enumerate(zip(signals, prices)):
        if signal == 'BUY' and capital > price:
            shares_to_buy = capital // price
            shares += shares_to_buy
            capital -= shares_to_buy * price

        elif signal == 'SELL' and shares > 0:
            capital += shares * price
            shares = 0

        portfolio_value = capital + shares * price
        portfolio_values.append(portfolio_value)

    total_return = (portfolio_values[-1] - initial_capital) / initial_capital * 100

    return portfolio_values, total_return


# ===============================================================================
# ADVANCED ANALYSIS FUNCTIONS
# ===============================================================================

class AdvancedAnalyzer(StockTimeSeriesAnalyzer):
    """
    Extended analyzer with advanced features
    """

    def __init__(self):
        super().__init__()
        self.technical_indicators = {}

    def calculate_technical_indicators(self, price_column='Close'):
        """
        Calculate technical indicators
        """
        print("\n📊 Calculating Technical Indicators...")

        data = self.processed_data[price_column]

        # Moving averages
        self.technical_indicators['SMA_10'] = data.rolling(window=10).mean()
        self.technical_indicators['SMA_20'] = data.rolling(window=20).mean()
        self.technical_indicators['EMA_10'] = data.ewm(span=10).mean()

        # RSI
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.technical_indicators['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = data.ewm(span=12).mean()
        ema_26 = data.ewm(span=26).mean()
        self.technical_indicators['MACD'] = ema_12 - ema_26
        self.technical_indicators['MACD_Signal'] = self.technical_indicators['MACD'].ewm(span=9).mean()

        # Bollinger Bands
        sma_20 = data.rolling(window=20).mean()
        std_20 = data.rolling(window=20).std()
        self.technical_indicators['BB_Upper'] = sma_20 + (std_20 * 2)
        self.technical_indicators['BB_Lower'] = sma_20 - (std_20 * 2)

        print("✅ Technical indicators calculated")

    def plot_technical_analysis(self, price_column='Close'):
        """
        Plot comprehensive technical analysis
        """
        if not self.technical_indicators:
            self.calculate_technical_indicators(price_column)

        fig, axes = plt.subplots(4, 1, figsize=(15, 20))
        fig.suptitle('Technical Analysis Dashboard', fontsize=16)

        # Price and Moving Averages
        axes[0].plot(self.processed_data.index, self.processed_data[price_column],
                    label='Close Price', linewidth=2)
        axes[0].plot(self.processed_data.index, self.technical_indicators['SMA_10'],
                    label='SMA 10', alpha=0.7)
        axes[0].plot(self.processed_data.index, self.technical_indicators['SMA_20'],
                    label='SMA 20', alpha=0.7)
        axes[0].fill_between(self.processed_data.index,
                           self.technical_indicators['BB_Upper'],
                           self.technical_indicators['BB_Lower'],
                           alpha=0.2, label='Bollinger Bands')
        axes[0].set_title('Price and Moving Averages')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # RSI
        axes[1].plot(self.processed_data.index, self.technical_indicators['RSI'])
        axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.7)
        axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.7)
        axes[1].set_title('RSI (Relative Strength Index)')
        axes[1].set_ylabel('RSI')
        axes[1].grid(True, alpha=0.3)

        # MACD
        axes[2].plot(self.processed_data.index, self.technical_indicators['MACD'],
                    label='MACD')
        axes[2].plot(self.processed_data.index, self.technical_indicators['MACD_Signal'],
                    label='Signal Line')
        axes[2].bar(self.processed_data.index,
                   self.technical_indicators['MACD'] - self.technical_indicators['MACD_Signal'],
                   alpha=0.3, label='Histogram')
        axes[2].set_title('MACD')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Volume
        if 'Volume' in self.processed_data.columns:
            axes[3].bar(self.processed_data.index, self.processed_data['Volume'],
                       alpha=0.7)
            axes[3].set_title('Volume')
            axes[3].set_ylabel('Volume')

        plt.tight_layout()
        plt.show()


# ===============================================================================
# PROJECT STRUCTURE GENERATOR
# ===============================================================================

def create_project_structure():
    """
    Create organized project structure
    """
    folders = [
        'data',
        'models',
        'results',
        'notebooks',
        'scripts',
        'reports'
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    print("📁 Project structure created:")
    for folder in folders:
        print(f"   └── {folder}/")

    # Create README
    readme_content = """
# Stock Market Time Series Analysis Project

## Project Structure
- `data/`: Raw and processed data files
- `models/`: Trained model files
- `results/`: Analysis results and predictions
- `notebooks/`: Jupyter notebooks for analysis
- `scripts/`: Python scripts
- `reports/`: Generated reports

## Usage
1. Place your Excel file in the `data/` folder
2. Run the main analysis script
3. Check results in `results/` and `reports/` folders

## Models Implemented
- ARIMA (AutoRegressive Integrated Moving Average)
- SARIMA (Seasonal ARIMA)
- Prophet (Facebook's forecasting model)
- LSTM (Long Short-Term Memory neural network)

## Requirements
See requirements in the main script or install via pip.
"""

    with open('README.md', 'w') as f:
        f.write(readme_content)

    print("📝 README.md created")


# Call this function to set up project structure
if __name__ == "__main__":
    create_project_structure()
