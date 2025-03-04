# GMF-Investments

## Overview
This project focuses on applying advanced time series forecasting techniques to optimize portfolio management. The objective is to analyze historical financial data, predict future market trends, and provide investment recommendations based on forecasted insights. The project utilizes Python, YFinance, and machine learning models such as ARIMA, SARIMA, and LSTM to enhance financial decision-making.

## Business Objective
Guide Me in Finance (GMF) Investments is a financial advisory firm specializing in personalized portfolio management. By integrating time series forecasting models, GMF aims to:
- Predict market trends with data-driven insights.
- Optimize asset allocation strategies.
- Enhance portfolio performance while minimizing risks.

## Data Description
The dataset consists of historical financial data for three key assets:
- **Tesla (TSLA)**: High-growth, high-risk stock in the automobile manufacturing sector.
- **Vanguard Total Bond Market ETF (BND)**: A bond ETF tracking U.S. investment-grade bonds, providing stability and income.
- **S&P 500 ETF (SPY)**: An ETF tracking the S&P 500 Index, offering broad market exposure.

Data is sourced from **YFinance** for the period **January 1, 2015 – January 31, 2025** and includes:
- **Date**: Trading day timestamp.
- **Open, High, Low, Close**: Daily price metrics.
- **Adj Close**: Adjusted close price (accounting for dividends/splits).
- **Volume**: The total number of shares/units traded each day.

## Project Tasks

### 1. Data Preprocessing & Exploration
- Extract historical financial data using YFinance.
- Clean and preprocess data:
  - Handle missing values.
  - Normalize or scale data where necessary.
  - Identify trends and patterns through exploratory data analysis (EDA).
- Perform volatility analysis:
  - Calculate rolling means and standard deviations.
  - Identify significant anomalies using outlier detection.

### 2. Time Series Forecasting Models
- Train and evaluate forecasting models:
  - **ARIMA**: Suitable for univariate time series without seasonality.
  - **SARIMA**: Extends ARIMA by incorporating seasonal trends.
  - **LSTM**: Deep learning-based recurrent neural network model.
- Split data into training and testing sets.
- Optimize model parameters using grid search.
- Evaluate model performance using:
  - **Mean Absolute Error (MAE)**
  - **Root Mean Squared Error (RMSE)**
  - **Mean Absolute Percentage Error (MAPE)**

### 3. Forecast Future Market Trends
- Generate 6-12 months of future price predictions.
- Visualize forecasts with confidence intervals.
- Interpret trends, risks, and market opportunities based on predictions.

### 4. Portfolio Optimization
- Construct a portfolio comprising TSLA, BND, and SPY.
- Forecast BND and SPY prices and integrate them with TSLA forecasts.
- Compute annual returns, risk (volatility), and Sharpe Ratio.
- Use covariance matrices to assess asset relationships.
- Optimize asset allocation using:
  - **Maximum Sharpe Ratio approach**
  - **Risk-adjusted portfolio rebalancing**
- Visualize portfolio performance, cumulative returns, and risk-return trade-offs.

## Technologies Used
- **Python**
- **YFinance** (Financial data extraction)
- **Pandas, NumPy** (Data preprocessing & analysis)
- **Matplotlib, Seaborn** (Data visualization)
- **Statsmodels** (ARIMA, SARIMA models)
- **TensorFlow/Keras** (LSTM model)
- **SciPy, CVXPY** (Portfolio optimization)

## Expected Outcomes
- **Competence in time series forecasting** using financial datasets.
- **Ability to analyze and model stock market trends** for investment strategies.
- **Portfolio optimization techniques** using predictive insights.
- **Risk assessment and return maximization** through quantitative models.

## Installation & Usage
To run the project locally, follow these steps:
1. Clone the repository:
   ```sh
   git clone https://github.com/Tewodros-agegnehu/GMF-Investments.git
   cd portfolio-forecasting
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the data extraction script:
   ```sh
   python data_extraction.py
   ```
4. Train forecasting models:
   ```sh
   python train_model.py
   ```
5. Generate forecasts and optimize the portfolio:
   ```sh
   python optimize_portfolio.py
   ```

## Project Structure
```
portfolio-forecasting/
│── data/
│   ├── historical_data.csv
│── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── forecasting_models.ipynb
│── src/
│   ├── data_extraction.py
│   ├── train_model.py
│   ├── optimize_portfolio.py
│── README.md
│── requirements.txt
```

## License
This project is open-source and available under the **MIT License**.

## Contact
For any inquiries or contributions, feel free to reach out to:
- **Tewodros Agegnehu**
- **Email:** tedyagegnehu@gmail.com
- **LinkedIn:** [Your LinkedIn Profile](https://linkedin.com/in/tewodrosagegnehu)

---

