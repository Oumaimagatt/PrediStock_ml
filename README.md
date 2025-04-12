# 📈 Masimo Corporation (MASI) Stock Analysis & Price Prediction App

Welcome to the **MASI Stock Analysis** app — a powerful and interactive Streamlit-based web application that enables you to visualize stock price trends, apply key technical indicators, and predict short-term stock prices using multiple machine learning models.

---

### 👨‍💻 Designed by:
- [Megzari Omar](https://www.linkedin.com/in/omar-megzari7/)
- [Gattoua Oumaima](https://www.linkedin.com/in/oumaima-gattoua/)
- [Hajji Malak](https://www.linkedin.com/in/malak-hajji-8048a6293/)

You can find more about me and my colleagues projects on our [GitHub profile](https://github.com/omarmegzari) (https://github.com/Oumaimagatt) (https://github.com/malakiies).

---

## 📚 Table of Contents
- [📌 Project Description](#project-description)
- [✨ Features](#features)
- [🚀 Getting Started](#getting-started)
- [🧠 Technologies Used](#technologies-used)
- [📬 Contact & More Projects](#contact--more-projects)

---

## 📌 Project Description

The **MASI Stock Analysis** app is a user-friendly web tool designed for investors, analysts, and enthusiasts to explore historical stock data, apply technical indicators, and generate short-term stock price forecasts using popular machine learning algorithms.

---

## ✨ Features

🔍 **Interactive Visualization**  
- Plot key technical indicators:
  - **Close Price** – The last traded price becomes the official closing price.
  - **MACD (Moving Average Convergence Divergence)**
  - **RSI (Relative Strength Index)**
  - **SMA (Simple Moving Average)**
  - **EMA (Exponential Moving Average)**
- Visualize stock closing prices over customizable date ranges.

📊 **Price Prediction**  
- Predict short-term stock prices (1 to 5 days ahead).
- Choose from various ML models:
  - Linear Regression  
  - Random Forest Regressor  
  - Extra Trees Regressor  
  - K-Nearest Neighbors  
  - XGBoost Regressor  
- Evaluate model performance with metrics like **R² Score** and **Mean Absolute Error (MAE)**.

💾 **Download the MASI Data**  
- Export historical stock data as a CSV file.

---

## 🚀 Getting Started

To run the app locally:

```bash
git clone https://github.com/Oumaimagatt/PrediStock_ml.git
cd PrediStock_ml
streamlit run app.py
# or
python -m streamlit run app.py
