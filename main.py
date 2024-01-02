# pip install streamlit fbprophet plotly pandas openpyxl
import streamlit as st
from datetime import date
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go



# Function to load data from an Excel file
@st.cache_data
def load_data_from_excel(file_path):
    data = pd.read_excel(file_path, engine='openpyxl')
    return data

# Function to format numeric values
def format_numeric_value(value):
    return f'{value:.2f}' if value % 1 != 0 else f'{int(value)}.00'


# Streamlit app title
st.title('Web Prediksi Harga Pokok Riau')

# Load data from an Excel file in the same directory
file_path = '/workspaces/2024/harga.xlsx'  # Replace with the actual name of your Excel file
data = load_data_from_excel(file_path)

# User selects price commodity dynamically
selected_price_commodity = st.selectbox('Pilih Komoditas Untuk Prediksi', data.columns[1:])
# Assuming the Date column is at index 0, so we start from index 1

# Rename columns in the forecast data
renamed_data = data.rename(columns={"Date": "Tanggal"})


# Display raw data for the selected column only
st.subheader(f'Data Harga {selected_price_commodity}')
raw_data_table = renamed_data[['Tanggal', selected_price_commodity]].sort_values(by='Tanggal', ascending=False)
raw_data_table['Tanggal'] = raw_data_table['Tanggal'].dt.date
raw_data_table[selected_price_commodity] = raw_data_table[selected_price_commodity].map(format_numeric_value)
st.write(raw_data_table)







# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data[selected_price_commodity], name=selected_price_commodity))
    fig.layout.update(title_text='Plot Data Harga', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

n_years = st.slider('Tahun Prediksi:', 1, 5)
period = n_years * 365

# Predict forecast with Prophet
df_train = data[['Date', selected_price_commodity]]
df_train = df_train.rename(columns={"Date": "ds", selected_price_commodity: "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Rename columns in the forecast data
renamed_forecast = forecast.rename(columns={"ds": "Tanggal", "yhat": "Harga Prediksi"})

# Show and plot forecast with renamed columns
st.subheader('Data Prediksi')
renamed_forecast['Tanggal'] = renamed_forecast['Tanggal'].dt.date
renamed_forecast['Harga Prediksi'] = renamed_forecast['Harga Prediksi'].map(format_numeric_value)
st.write(renamed_forecast[['Tanggal', 'Harga Prediksi']].sort_values(by='Tanggal', ascending=False))


st.write(f'Plot Prediksi Harga Untuk {n_years} Tahun')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

#st.write("Komponen Prediksi")
#fig2 = m.plot_components(forecast)
#st.write(fig2)
