# Cell 1: Setup and Imports
# Enhanced MKSI Stock Valuation Model - Local Version

# Direct imports (assume user installs packages via requirements.txt)
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings
import time
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')
# %matplotlib inline  # Removed for local use

print("Cell 1: Environment setup complete.")

# Cell 2: Model Class Definition
class MKSIValuationModel:
    def __init__(self, start_date="2015-01-01", end_date=None):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.model = None
        self.data = None
        self.feature_importance = None
        self.validation_metrics = None
        self.feature_cols = []

    def fetch_data_safely(self, source, symbol, start, end, max_retries=3):
        """Robust data fetching with error handling"""
        for attempt in range(max_retries):
            try:
                print(f"Fetching {symbol} from {source} (attempt {attempt+1})")
                if source == "yfinance":
                    data = yf.download(symbol, start=start, end=end, progress=False)
                    if not data.empty:
                        # Handle different column formats from yfinance
                        if isinstance(data.columns, pd.MultiIndex):
                            # Multi-level columns (normal case)
                            return data
                        else:
                            # Single-level columns - convert to expected format
                            if 'Close' in data.columns and 'Adj Close' not in data.columns:
                                data['Adj Close'] = data['Close']
                            return data
                elif source == "fred":
                    data = pdr.DataReader(symbol, "fred", start, end)
                    if not data.empty:
                        return data

                print(f"No data returned for {symbol}")
                time.sleep(1)

            except Exception as e:
                print(f"Attempt {attempt+1} failed for {symbol}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"Failed to fetch {symbol} after {max_retries} attempts")
                    return pd.DataFrame()

        return pd.DataFrame()

    def get_real_fundamentals(self):
        """Extract real MKSI fundamental data"""
        try:
            ticker = yf.Ticker("MKSI")
            financials_q = ticker.quarterly_financials
            balance_q = ticker.quarterly_balance_sheet
            cashflow_q = ticker.quarterly_cashflow

            if financials_q.empty:
                print("Warning: No fundamental data available, using synthetic data")
                return self._generate_synthetic_fundamentals()

            funds_data = []
            for date in financials_q.columns:
                try:
                    revenue = financials_q.loc['Total Revenue', date] if 'Total Revenue' in financials_q.index else np.nan
                    ebit = financials_q.loc['Operating Income', date] if 'Operating Income' in financials_q.index else np.nan
                    capex = abs(cashflow_q.loc['Capital Expenditure', date]) if 'Capital Expenditure' in cashflow_q.index else np.nan
                    net_income = financials_q.loc['Net Income', date] if 'Net Income' in financials_q.index else np.nan
                    total_assets = balance_q.loc['Total Assets', date] if 'Total Assets' in balance_q.index else np.nan

                    funds_data.append({
                        'date': date,
                        'revenue': revenue / 1e6 if pd.notna(revenue) else np.nan,
                        'ebit': ebit / 1e6 if pd.notna(ebit) else np.nan,
                        'capex': capex / 1e6 if pd.notna(capex) else np.nan,
                        'nopat': net_income / 1e6 if pd.notna(net_income) else np.nan,
                        'invested_capital': total_assets / 1e6 if pd.notna(total_assets) else np.nan
                    })

                except KeyError as e:
                    print(f"Missing data for {date}: {e}")
                    continue

            if not funds_data:
                print("No usable fundamental data found, using synthetic data")
                return self._generate_synthetic_fundamentals()

            funds_df = pd.DataFrame(funds_data)
            funds_df.set_index('date', inplace=True)
            funds_df.sort_index(inplace=True)
            return funds_df

        except Exception as e:
            print(f"Error fetching fundamentals: {e}")
            return self._generate_synthetic_fundamentals()

    def _generate_synthetic_fundamentals(self):
        """Generate realistic synthetic fundamental data as fallback"""
        print("Generating synthetic fundamental data...")
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='Q')
        n_periods = len(dates)
        trend = np.linspace(0, 0.3, n_periods)
        noise = np.random.normal(0, 0.1, n_periods)

        funds_df = pd.DataFrame({
            'revenue': 800 * (1 + trend + noise * 0.2),
            'ebit': 150 * (1 + trend + noise * 0.3),
            'capex': 40 * (1 + trend * 0.5 + noise * 0.4),
            'nopat': 120 * (1 + trend + noise * 0.25),
            'invested_capital': 1000 * (1 + trend * 0.6 + noise * 0.15)
        }, index=dates)

        return funds_df

    def calculate_rolling_beta(self, stock_returns, market_returns, window=8):
        """Robust rolling beta calculation with validation"""
        betas = []
        for i in range(len(stock_returns)):
            if i < window - 1:
                betas.append(np.nan)
                continue

            stock_slice = stock_returns.iloc[i-window+1:i+1].dropna()
            market_slice = market_returns.iloc[i-window+1:i+1].dropna()

            if len(stock_slice) < 4 or len(market_slice) < 4:
                betas.append(np.nan)
                continue

            combined = pd.concat([stock_slice, market_slice], axis=1).dropna()
            if len(combined) < 4:
                betas.append(np.nan)
                continue

            try:
                covariance = np.cov(combined.iloc[:, 0], combined.iloc[:, 1])[0, 1]
                market_variance = np.var(combined.iloc[:, 1])

                if market_variance > 0:
                    beta = covariance / market_variance
                    beta = max(-3, min(3, beta))
                    betas.append(beta)
                else:
                    betas.append(np.nan)
            except Exception:
                betas.append(np.nan)

        return pd.Series(betas, index=stock_returns.index)

    def create_technical_features(self, price_data):
        """Create technical indicators"""
        if isinstance(price_data, pd.DataFrame):
            if 'Adj Close' in price_data.columns:
                price_data = price_data['Adj Close']
            elif 'Close' in price_data.columns:
                price_data = price_data['Close']
            else:
                price_data = price_data.iloc[:, 0]

        features = pd.DataFrame(index=price_data.index)
        features['sma_20'] = price_data.rolling(20).mean()
        features['sma_50'] = price_data.rolling(50).mean()
        features['price_to_sma20'] = price_data / features['sma_20']

        returns = price_data.pct_change()
        features['volatility_20'] = returns.rolling(20).std()
        features['volatility_60'] = returns.rolling(60).std()
        features['momentum_20'] = price_data / price_data.shift(20) - 1
        features['momentum_60'] = price_data / price_data.shift(60) - 1

        delta = price_data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))

        return features

    def create_lag_features(self, df, columns, lags=[1, 2, 4]):
        """Create lagged versions of fundamental metrics"""
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f"{col}_lag{lag}"] = df[col].shift(lag)
        return df

    def collect_data(self):
        """Collect all required data sources with robust error handling"""
        print("Starting data collection...")
        
        funds_q = self.get_real_fundamentals()
        funds_q["rev_growth_yoy"] = funds_q["revenue"].pct_change(4)
        funds_q["ebit_margin"] = funds_q["ebit"] / funds_q["revenue"]
        funds_q["capex_pct"] = funds_q["capex"] / funds_q["revenue"]
        funds_q["roic"] = funds_q["nopat"] / funds_q["invested_capital"]

        fundamental_cols = ["rev_growth_yoy", "ebit_margin", "roic"]
        funds_q = self.create_lag_features(funds_q, fundamental_cols, lags=[1, 2])

        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)

        print("Fetching macro data...")
        treasury_data = self.fetch_data_safely("fred", "DGS10", start_date, end_date)
        if not treasury_data.empty:
            funds_q["10y_rate"] = treasury_data.resample("Q").last().reindex(funds_q.index) / 100
        else:
            funds_q["10y_rate"] = 0.03

        pmi_data = self.fetch_data_safely("fred", "MANEMP", start_date, end_date)
        if not pmi_data.empty:
            funds_q["pmi"] = pmi_data.resample("Q").last().reindex(funds_q.index)
        else:
            funds_q["pmi"] = 50

        print("Fetching market data...")
        sox_data = self.fetch_data_safely("yfinance", "^SOX", start_date, end_date)
        if not sox_data.empty:
            close_col = 'Adj Close' if 'Adj Close' in sox_data.columns else 'Close'
            if close_col in sox_data.columns:
                sox_quarterly = sox_data[close_col].resample("Q").last()
                funds_q["sox_ret"] = sox_quarterly.pct_change().reindex(funds_q.index)
            else:
                funds_q["sox_ret"] = 0
        else:
            funds_q["sox_ret"] = 0

        vix_data = self.fetch_data_safely("yfinance", "^VIX", start_date, end_date)
        if not vix_data.empty:
            close_col = 'Adj Close' if 'Adj Close' in vix_data.columns else 'Close'
            if close_col in vix_data.columns:
                funds_q["vix"] = vix_data[close_col].resample("Q").last().reindex(funds_q.index)
            else:
                funds_q["vix"] = 20
        else:
            funds_q["vix"] = 20

        mksi_data = self.fetch_data_safely("yfinance", "MKSI", start_date, end_date)
        spy_data = self.fetch_data_safely("yfinance", "SPY", start_date, end_date)

        if not mksi_data.empty and not spy_data.empty:
            mksi_close = 'Adj Close' if 'Adj Close' in mksi_data.columns else 'Close'
            spy_close = 'Adj Close' if 'Adj Close' in spy_data.columns else 'Close'

            if mksi_close in mksi_data.columns and spy_close in spy_data.columns:
                mksi_returns = mksi_data[mksi_close].pct_change()
                spy_returns = spy_data[spy_close].pct_change()
                mksi_q_returns = mksi_returns.resample("Q").apply(lambda x: (1 + x).prod() - 1)
                spy_q_returns = spy_returns.resample("Q").apply(lambda x: (1 + x).prod() - 1)
                funds_q["beta"] = self.calculate_rolling_beta(mksi_q_returns, spy_q_returns)

                tech_features = self.create_technical_features(mksi_data[mksi_close])
                tech_quarterly = tech_features.resample("Q").last()

                for col in ['price_to_sma20', 'volatility_20', 'momentum_20', 'rsi']:
                    if col in tech_quarterly.columns:
                        funds_q[col] = tech_quarterly[col].reindex(funds_q.index)

                funds_q["price"] = mksi_data[mksi_close].resample("Q").last().reindex(funds_q.index)
            else:
                print("Warning: Required price columns not found, using defaults")
                self._set_default_market_data(funds_q)
        else:
            print("Warning: Could not fetch MKSI/SPY data, using defaults")
            self._set_default_market_data(funds_q)

        # Store the complete dataset
        self.data = funds_q.dropna(subset=['price'])
        # Fixed deprecated method calls
        self.data = self.data.ffill().bfill()
        print(f"Data collection complete. Shape: {self.data.shape}")

        self.feature_cols = [
            "rev_growth_yoy", "ebit_margin", "capex_pct", "roic",
            "rev_growth_yoy_lag1", "ebit_margin_lag1", "roic_lag1",
            "10y_rate", "pmi", "sox_ret", "vix", "beta",
            "price_to_sma20", "volatility_20", "momentum_20", "rsi"
        ]

        self.feature_cols = [col for col in self.feature_cols if col in self.data.columns]
        print(f"Features available: {len(self.feature_cols)}")
        return self.data

    def _set_default_market_data(self, funds_q):
        """Set default market data when API calls fail"""
        funds_q["beta"] = 1.0
        funds_q["price_to_sma20"] = 1.0
        funds_q["volatility_20"] = 0.02
        funds_q["momentum_20"] = 0.0
        funds_q["rsi"] = 50
        funds_q["price"] = 100

    def comprehensive_model_validation(self, X, y, model, cv_splits=5):
        """Enhanced cross-validation with multiple metrics"""
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        metrics = {'mae': [], 'mse': [], 'r2': [], 'mape': []}
        predictions = []
        actuals = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"Validating fold {fold + 1}/{cv_splits}")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics['mae'].append(mean_absolute_error(y_test, y_pred))
            metrics['mse'].append(mean_squared_error(y_test, y_pred))
            metrics['r2'].append(r2_score(y_test, y_pred))
            metrics['mape'].append(mean_absolute_percentage_error(y_test, y_pred))

            predictions.extend(y_pred)
            actuals.extend(y_test)

        summary = {}
        for metric, values in metrics.items():
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)

        summary['rmse_mean'] = np.sqrt(summary['mse_mean'])
        return summary, predictions, actuals

    def train_model(self):
        """Train and validate the model with comprehensive evaluation"""
        if self.data is None:
            raise ValueError("No data available. Run collect_data() first.")

        print("Starting model training...")
        X = self.data[self.feature_cols]
        y = self.data["price"]
        print(f"Training with {len(self.feature_cols)} features and {len(X)} samples")

        tscv = TimeSeriesSplit(n_splits=5)
        rf = RandomForestRegressor(random_state=42)

        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "min_samples_leaf": [3, 5, 10],
            "min_samples_split": [5, 10, 15]
        }

        print("Running grid search...")
        gsearch = GridSearchCV(
            rf, param_grid, cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=2, verbose=1
        )

        gsearch.fit(X, y)
        self.model = gsearch.best_estimator_
        print(f"Best parameters: {gsearch.best_params_}")

        print("Running comprehensive validation...")
        self.validation_metrics, predictions, actuals = self.comprehensive_model_validation(X, y, self.model)

        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_cols
        ).sort_values(ascending=False)

        print("\n" + "="*50)
        print("MODEL VALIDATION RESULTS")
        print("="*50)
        for metric, value in self.validation_metrics.items():
            if 'mean' in metric:
                print(f"{metric.upper()}: {value:.4f}")

        print("\nTOP 10 FEATURE IMPORTANCES:")
        print(self.feature_importance.head(10))
        return self.model

    def generate_correlated_scenarios(self, n_sims=1000):
        """Generate scenarios preserving historical correlations"""
        if self.data is None:
            raise ValueError("No data available. Run collect_data() first.")

        feature_data = self.data[self.feature_cols].dropna()
        if len(feature_data) < 10:
            print("Insufficient data for correlation analysis, using independent scenarios")
            return self._generate_independent_scenarios(n_sims)

        try:
            mean_vals = feature_data.mean()
            cov_matrix = feature_data.cov()
            if np.isscalar(cov_matrix):
                cov_matrix = np.array([[cov_matrix]])
            else:
                cov_matrix = cov_matrix.values
            if cov_matrix.shape[0] == cov_matrix.shape[1]:
                cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6

            scenarios = np.random.multivariate_normal(
                mean=np.atleast_1d(mean_vals),  # Ensure 1D numpy array robustly
                cov=cov_matrix,
                size=n_sims
            )
            # Ensure scenarios is 2D and columns match, and columns are strings
            return pd.DataFrame(scenarios, columns=pd.Index(self.feature_cols))

        except Exception as e:
            print(f"Error generating correlated scenarios: {e}")
            return self._generate_independent_scenarios(n_sims)

    def _generate_independent_scenarios(self, n_sims):
        """Fallback independent scenario generation"""
        if self.data is None:
            raise ValueError("No data available. Run collect_data() first.")
        scenarios = pd.DataFrame()
        for col in self.feature_cols:
            if col in self.data.columns:
                mean_val = self.data[col].mean()
                std_val = self.data[col].std()
                scenarios[col] = np.random.normal(mean_val, std_val, n_sims)
            else:
                scenarios[col] = np.zeros(n_sims)
        return scenarios

    def run_simulation(self, n_sims=1000):
        """Run Monte Carlo simulation with correlated scenarios"""
        if self.model is None:
            raise ValueError("No trained model available. Run train_model() first.")
        if self.data is None:
            raise ValueError("No data available. Run collect_data() first.")

        print(f"Running Monte Carlo simulation with {n_sims} scenarios...")
        scenarios = self.generate_correlated_scenarios(n_sims)
        sim_preds = self.model.predict(scenarios)

        results = {
            'mean': np.mean(sim_preds),
            'median': np.median(sim_preds),
            'std': np.std(sim_preds),
            'p5': np.percentile(sim_preds, 5),
            'p25': np.percentile(sim_preds, 25),
            'p75': np.percentile(sim_preds, 75),
            'p95': np.percentile(sim_preds, 95),
            'current_price': self.data['price'].iloc[-1] if len(self.data) > 0 else 100
        }

        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.hist(sim_preds, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(results['mean'], color='red', linestyle='--', label=f"Mean: ${results['mean']:.2f}")
        plt.axvline(results['current_price'], color='green', linestyle='--', label=f"Current: ${results['current_price']:.2f}")
        plt.title('Monte Carlo Fair Value Distribution')
        plt.xlabel('Predicted Fair Value ($)')
        plt.ylabel('Frequency')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.boxplot(sim_preds)
        plt.title('Fair Value Box Plot')
        plt.ylabel('Price ($)')

        plt.subplot(2, 2, 3)
        if hasattr(self, 'feature_importance') and self.feature_importance is not None:
            top_features = self.feature_importance.head(8)
            plt.barh(range(len(top_features)), top_features.values.astype(float))  # Ensure float array
            plt.yticks(range(len(top_features)), list(map(str, top_features.index.tolist())))  # Ensure string labels
            plt.title('Top Feature Importances')
            plt.xlabel('Importance')

        plt.subplot(2, 2, 4)
        if hasattr(self, 'validation_metrics'):
            actual_prices = self.data['price'].values.astype(float)
            predicted_prices = self.model.predict(self.data[self.feature_cols]).astype(float)
            plt.scatter(actual_prices, predicted_prices, alpha=0.6)
            plt.plot([actual_prices.min(), actual_prices.max()],
                    [actual_prices.min(), actual_prices.max()], 'r--')
            plt.xlabel('Actual Price ($)')
            plt.ylabel('Predicted Price ($)')
            plt.title('Actual vs Predicted')

        plt.tight_layout()
        plt.show()

        print("\n" + "="*50)
        print("MONTE CARLO SIMULATION RESULTS")
        print("="*50)
        print(f"Mean Fair Value:    ${results['mean']:.2f}")
        print(f"Median Fair Value:  ${results['median']:.2f}")
        print(f"Current Price:      ${results['current_price']:.2f}")
        print(f"Upside Potential:   {((results['mean']/results['current_price'])-1)*100:.1f}%")
        print(f"\nConfidence Intervals:")
        print(f"5th percentile:     ${results['p5']:.2f}")
        print(f"25th percentile:    ${results['p25']:.2f}")
        print(f"75th percentile:    ${results['p75']:.2f}")
        print(f"95th percentile:    ${results['p95']:.2f}")

        return results, sim_preds

# Cell 3: Main Execution (FIXED - moved outside class)
if __name__ == "__main__":
    print("🚀 Enhanced MKSI Valuation Model - Local Version")
    print("="*60)

    # Initialize model
    model = MKSIValuationModel(start_date="2021-01-01")

    print("📊 Step 1: Collecting data...")
    data = model.collect_data()

    print("🤖 Step 2: Training model...")
    trained_model = model.train_model()

    print("🎲 Step 3: Running Monte Carlo simulation...")
    results, predictions = model.run_simulation(n_sims=500)

    print("\n✅ Model training and simulation complete!")
    print("📈 Check the plots above for detailed results!")

    print("\n" + "="*60)
    print("📋 SUMMARY RESULTS")
    print("="*60)
    if model.validation_metrics is not None and 'r2_mean' in model.validation_metrics:
        print(f"📊 Model Performance (R²): {model.validation_metrics['r2_mean']:.3f}")
    else:
        print("📊 Model Performance (R²): N/A")
    print(f"💰 Fair Value Estimate: ${results['mean']:.2f}")
    print(f"📈 Current Price: ${results['current_price']:.2f}")
    print(f"🎯 Upside Potential: {((results['mean']/results['current_price'])-1)*100:.1f}%")
    print("="*60)