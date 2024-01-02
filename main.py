import streamlit as st
from datetime import date
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Function to load data from an Excel file
@st.cache_data
def load_data_from_excel(file_path):
    data = pd.read_excel(file_path)
    return data

# Streamlit app title
st.title('Stock Forecast App')

# Load data from an Excel file in the same directory
file_path = '/workspaces/2024/harga.csv'  # Replace with the actual name of your Excel file
data = load_data_from_excel(file_path)

# User selects price commodity dynamically
selected_price_commodity = st.selectbox('Select price commodity for prediction', data.columns[1:])
# Assuming the Date column is at index 0, so we start from index 1

# Display raw data
st.subheader('Raw data')
st.write(data.sort_values(by='Date', ascending=False))

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data[selected_price_commodity], name=selected_price_commodity))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Predict forecast with Prophet
df_train = data[['Date', selected_price_commodity]]
df_train = df_train.rename(columns={"Date": "ds", selected_price_commodity: "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=365)  # Assuming 1 year forecast
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.sort_values(by='ds', ascending=False))

st.write(f'Forecast plot for 1 year')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
