
import streamlit as st
import pandas as pd
import yfinance as yf
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Page configuration
st.set_page_config(
    page_title="MASI Stock Analysis",
    page_icon="📈",
    layout="wide"
)

# --- Welcome Screen Logic ---
if 'show_main' not in st.session_state:
    st.session_state.show_main = False

if not st.session_state.show_main:
    st.title("📈 Welcome to Masimo Corporation (MASI) Stock Analysis")

    st.markdown("<br>", unsafe_allow_html=True)

    st.write("Dive into powerful data visualizations and smart predictions to understand and forecast the performance of Masimo Corporation (MASI) stock")

    if st.button("Get Started"):
        st.session_state.show_main = True
        st.rerun()
else:
    st.title('Masimo Corporation (MASI) Stock Analysis')
    st.sidebar.title('SPECIFY DATE PARAMETERS TO GENERATE INSIGHTS OR PROJECTIONS')

    def main():
        option = st.sidebar.selectbox('Select Analysis Type', ['Select', 'Visualization', 'Price Prediction'])
        if option == 'Visualization':
            tech_indicators()
        elif option == 'Price Prediction':
            predict()
        elif option == 'Select':
            st.sidebar.info("Please select an analysis type from the sidebar to proceed.")
            return

    @st.cache_resource
    def download_data(start_date, end_date):
        df = yf.download("MASI", start=start_date, end=end_date, progress=False)
        return df

    # Set default date range (3 years back)
    today = datetime.date.today()
    before = today - datetime.timedelta(days=1095)
    start_date = st.sidebar.date_input('Start Date', value=before)
    end_date = st.sidebar.date_input('End Date', today)

    if start_date < end_date:
        st.sidebar.success(f'Data range: {start_date} to {end_date}')
    else:
        st.sidebar.error('Error: End date must be after start date')

    data = download_data(start_date, end_date)
    scaler = StandardScaler()

    def tech_indicators():
        st.header('MASI Visualization')
        option = st.radio('Choose Indicator', ['Close Price', 'MACD', 'RSI', 'SMA', 'EMA'])

        close_prices = data['Close'].squeeze() if isinstance(data['Close'], pd.DataFrame) else data['Close']

        fig, ax = plt.subplots(figsize=(12, 6))

        if option == 'Close Price':
            sns.lineplot(data=close_prices, ax=ax, color='blue', label='Close Price')
            ax.set_title('MASI Closing Price', fontsize=16)
        elif option == 'MACD':
            macd_line = MACD(close_prices).macd()
            signal_line = MACD(close_prices).macd_signal()
            sns.lineplot(data=macd_line, ax=ax, color='blue', label='MACD Line')
            sns.lineplot(data=signal_line, ax=ax, color='red', label='Signal Line')
            ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax.set_title('MASI MACD', fontsize=16)
        elif option == 'RSI':
            rsi = RSIIndicator(close_prices).rsi()
            sns.lineplot(data=rsi, ax=ax, color='purple', label='RSI')
            ax.axhline(70, color='red', linestyle='--', label='Overbought (70)')
            ax.axhline(30, color='green', linestyle='--', label='Oversold (30)')
            ax.set_title('MASI Relative Strength Index', fontsize=16)
            ax.set_ylim(0, 100)
        elif option == 'SMA':
            sma = SMAIndicator(close_prices, window=14).sma_indicator()
            sns.lineplot(data=close_prices, ax=ax, color='blue', label='Close Price')
            sns.lineplot(data=sma, ax=ax, color='orange', label='SMA (14)')
            ax.set_title('MASI Simple Moving Average', fontsize=16)
        else:  # EMA
            ema = EMAIndicator(close_prices).ema_indicator()
            sns.lineplot(data=close_prices, ax=ax, color='blue', label='Close Price')
            sns.lineplot(data=ema, ax=ax, color='green', label='EMA')
            ax.set_title('MASI Exponential Moving Average', fontsize=16)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price/Value', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()
        plt.tight_layout()

        st.pyplot(fig)

    def predict():
        st.header('MASI Price Prediction')
        model = st.selectbox('Choose Prediction Model', 
                            ['Linear Regression', 'Random Forest', 'Extra Trees', 'K-Neighbors', 'XGBoost'])
        num = st.slider('Days to Predict', 1, 5, 1)

        if st.button('🚀 Generate Prediction'):
            with st.spinner('⏳ Training the model... please wait'):
                if model == 'Linear Regression':
                    engine = LinearRegression()
                elif model == 'Random Forest':
                    engine = RandomForestRegressor()
                elif model == 'Extra Trees':
                    engine = ExtraTreesRegressor()
                elif model == 'K-Neighbors':
                    engine = KNeighborsRegressor()
                else:
                    engine = XGBRegressor()

                df = data[['Close']]
                df['preds'] = data.Close.shift(-num)
                x = df.drop(['preds'], axis=1).values
                x = scaler.fit_transform(x)
                x_forecast = x[-num:]
                x = x[:-num]
                y = df.preds.values[:-num]

                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
                start_time = time.time()
                engine.fit(x_train, y_train)
                elapsed_time = time.time() - start_time

            st.success('✅ Model trained successfully!')
            st.info(f"🕒 Training completed in {elapsed_time:.2f} seconds")

            preds = engine.predict(x_test)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("R2 Score", f"{r2_score(y_test, preds):.4f}")
            with col2:
                st.metric("MAE", f"{mean_absolute_error(y_test, preds):.4f}")

            st.subheader(f"{num}-Day Price Forecast")
            forecast_pred = engine.predict(x_forecast)
            for i in range(num):
                st.write(f"📅 Day {i+1}: **${forecast_pred[i]:.2f}**")

    def save_to_csv(data):
        filename = "masi_data.csv"
        data.to_csv(filename, index=True)
        st.success(f"✅ Data saved as '{filename}' in your working directory!")

    if st.sidebar.button("💾 Save Data as CSV"):
        save_to_csv(data)

    if __name__ == '__main__':
        main()
    