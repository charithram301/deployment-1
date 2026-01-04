import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

data_path = r"C:\Users\chari\OneDrive\P587 DATASET.csv"

st.set_page_config(page_title="SARIMA Quick Deploy", layout="wide")

st.title("Stock Price Forecasting with SARIMA")
st.caption("Predicting future stock movements using time series modeling")

df=pd.read_csv(data_path)

st.subheader("Data Preview")
st.caption("Apple stocks")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
st.dataframe(df.head(10))
#converting data time index
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

st.subheader("Target series")
st.caption("close is the target variable")
series = df["Close"].dropna()
st.line_chart(series)

st.sidebar.header("SARIMA Hyperparameters")
p = st.sidebar.number_input("p (AR order)", min_value=0, max_value=5, value=1, step=1)
d = st.sidebar.number_input("d (difference order)", min_value=0, max_value=2, value=0, step=1)
q = st.sidebar.number_input("q (MA order)",min_value=0,max_value=2, value=0, step=1)
s = st.sidebar.number_input("s (seasonal period)",min_value=0, max_value=365, value=12, step=1)

train_size = st.sidebar.slider("Training proportion", min_value=0.5, max_value=0.95, value=0.8)
if st.button("Train SARIMA"):
    with st.spinner("Training SARIMA... this may take a while"):
        # split
        n_train = int(len(df) * float(train_size))
        train = series.iloc[:n_train]
        test = series.iloc[n_train:]
        try:
            model = SARIMAX(train, order=(p,d,q), seasonal_order=(0,0,0,s))
            res = model.fit(disp=False)
            pred=res.forecast(steps=len(test))
        except Exception as e:
            st.error(f"Model training failed: {e}")
            st.stop()
        st.success("Model trained")
        st.subheader("Model summary")
        st.text(res.summary().as_text())
        rmse = np.sqrt(mean_squared_error(test,pred))
        st.write("rmse:",rmse)
        steps = st.sidebar.number_input("Forecast steps", min_value=1, max_value=365, value=30, step=1)
        forecast = res.get_forecast(steps=steps)
        forecast_mean = forecast.predicted_mean
        # plot
        future_index=pd.date_range(start=series.index[-1], periods=steps+1, freq='M')[1:]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train.index, train, label="Train")
        ax.plot(test.index, test, label="Test")
        ax.plot(future_index, forecast_mean, label="Forecast")
        
        ax.legend()
        st.pyplot(fig)

st.subheader("Intractive Dashboard")

tableau_url = "https://public.tableau.com/views/stockpricedashboard1/Dashboard1?:showVizHome=no"

st.components.v1.iframe(
    tableau_url,
    width=1200,
    height=800,
    scrolling=True
)
       

        




