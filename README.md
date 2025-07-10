# 📈 Stock Portfolio Prediction with LSTM and PCA

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat\&logo=python\&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat\&logo=tensorflow\&logoColor=white)](https://www.tensorflow.org/)
[![yfinance](https://img.shields.io/badge/yfinance-1A1A1A?style=flat)](https://github.com/ranaroussi/yfinance)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat\&logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat\&logo=matplotlib\&logoColor=white)](https://matplotlib.org/)

---

## 🔍 Overview

This project implements a Long Short-Term Memory (LSTM) neural network model to predict the weekly average value of a stock portfolio composed of multiple tickers (e.g., AAPL, MSFT, GOOGL). The model uses historical weekly closing prices from the last 3 years to forecast future portfolio values several weeks ahead.

Additionally, Principal Component Analysis (PCA) is applied to analyze the variance and contribution of each stock within the portfolio.

---

## ⚙️ Features

* Download weekly historical closing prices of specified stocks via `yfinance`
* Compute weekly portfolio average price
* Data preprocessing with z-score normalization and sliding window approach
* Train and save an LSTM model to predict portfolio prices
* Model evaluation with multiple regression metrics (Huber loss, MSE, MAE, MAPE)
* Visualization of real vs predicted prices for test data and entire dataset
* Simple buy/hold/sell decision-making simulation based on model predictions
* Future forecasting for multiple weeks ahead
* PCA analysis to understand stock influence and variance contributions

---

## 📦 Requirements

* Python 3.8+
* TensorFlow
* yfinance
* pandas
* numpy
* matplotlib
* scikit-learn

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone <GIT-REPO-HTTPS>
cd repo-name
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install the dependencies:

```bash
pip install tensorflow yfinance pandas numpy matplotlib scikit-learn
```

4. Run the application:

```bash
python script_name.py
```

---

## 🚀 Usage

1. **Configure tickers and parameters**
   Edit the `TICKERS` list and other variables such as `WINDOW_SIZE`, `EPOCHS`, `BATCH_SIZE`, and `FUTURE_STEPS` as needed.
   
2. **Run the script/notebook**
   The script will:

   * Download historical weekly data (last 3 years)
   * Create and preprocess the portfolio dataset
   * Train the LSTM model (or load saved model if available)
   * Evaluate and plot predictions vs real data
   * Simulate buy/hold/sell decisions based on prediction changes
   * Forecast future portfolio values
   * Perform PCA to analyze stock contributions

3. **View results**
   Plots and printed output will show prediction accuracy, portfolio simulation results, and PCA insights.

---

## 📁 Project Structure

```
PrevisaoAcoes.ipynb               # Main notebook/script with all logic
modelo_lstm_portfolio_acoes.h5    # Saved LSTM model (auto-generated)
history_lstm.json                 # Training history (auto-generated)
```

---

## ⚙️ Model Details

* **Model architecture:**
  LSTM layer (64 units, tanh activation) → Dropout → Dense (64 units, ReLU) → Dropout → Dense (64 units, ELU) → Dropout → Dense (1 output)

* **Loss function:**
  Huber loss with δ=0.8 (robust regression loss combining MSE and MAE benefits)

* **Training:**
  Batch size = 4, Epochs = 300 (configurable)

---

## 📊 Buy/Hold/Sell Decision Logic

* The model predicts weekly portfolio values.
* A threshold percentage (default 1.5%) determines if the model suggests to buy, hold, or sell.
* The simulation starts with an initial capital (default \$1000).
* Buys and sells are made according to predicted price changes and available capital.

---

## 🔬 PCA Analysis

* PCA is performed on the scaled stock prices (excluding the portfolio average).
* The explained variance ratio shows how much each principal component captures of total variance.
* Component weights reveal the influence of each stock in principal components.
* Interpretation of principal components helps understand stock relationships and portfolio structure.

---

## 📈 Visualization

* Real vs predicted portfolio values for test set and entire dataset
* Future forecast for configurable number of weeks
* Training and validation loss curves
* PCA component contributions and explained variance

---

## 📚 References

* [TensorFlow LSTM tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
* [yfinance GitHub](https://github.com/ranaroussi/yfinance)
* [PCA explanation - scikit-learn](https://scikit-learn.org/stable/modules/decomposition.html#pca)

---

Note: The document contains the main analyses only. The code also includes: Buy/Hold/Sell decision suggestions based on predictions, LSTM model performance over the 3 years, and analysis of the second principal component (PC2).

