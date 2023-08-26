import time
import streamlit as st
import streamlit.components.v1 as comp 

from models.WavePattern import WavePattern
from models.WaveRules import Impulse, LeadingDiagonal, Correction
from models.WaveAnalyzer import WaveAnalyzer
from models.WaveCycle import WaveCycle
from models.WaveOptions import WaveOptionsGenerator5, WaveOptionsGenerator3
from models.helpers import plot_pattern
from models.helpers import plot_cycle
from models.helpers import plot_pattern_two

import plotly.graph_objects as go
import pandas as pd
import numpy as np

# HTML Heading Component 

html_heading = """
<link rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Crimson+Pro">
<style>
h2 {
    font-family: 'Crimson Pro', serif;
    font-size: 48px;
    text-align: center;
}
</style>

<h2> Asset Price Predictor </h2>

"""


# Heading 

with st.container():
    comp.html(html_heading, height=130)

# Asset Data

global df
df_data = pd.read_csv(r'./data/BTC-USD.csv') # 5 years
df = df_data.iloc[len(list(df_data.Date)) - 365 :, :] # 1 year plotting
idx_start = np.argmin(np.array(list(df['Low'])))

# """
# Wave Analyser Segment
# """

wa = WaveAnalyzer(df=df, verbose=False)
wave_options_impulse = WaveOptionsGenerator5(up_to=15)  # generates WaveOptions up to [15, 15, 15, 15, 15]
wave_options_correction = WaveOptionsGenerator3(up_to=9) 

impulse = Impulse('impulse')
leading_diagonal = LeadingDiagonal('leading diagonal')
correction = Correction('correction')
rules_to_check = [impulse, correction]

wavepatterns_up = list()
wavepatterns_down = list()
completeList = list()

# Check Impulsive Plotting

try:
    with st.spinner('#### **Impulsive Pattern Plotting in progress**'):
        imp_plot_find = st.empty()
        imp_plot_find.write(f'##### **Start index from the data at: {idx_start}**')
        
        for new_option_impulse in wave_options_impulse.options_sorted:
            waves_up = wa.find_impulsive_wave(idx_start=idx_start, wave_config=new_option_impulse.values)
            if waves_up:
                wavepattern_up = WavePattern(waves_up, verbose=True)

                for rule in rules_to_check:

                    if wavepattern_up.check_rule(rule):
                        if wavepattern_up in wavepatterns_up:
                                continue
                        else:
                            wavepatterns_up.append(wavepattern_up)
                        
                        imp_plot_find.write(f'##### **Impulsive Pattern found at: {new_option_impulse.values}**')
                        
                        cor_end = waves_up[4].idx_end
                        cor_date = waves_up[4].date_end
                        cor_close = waves_up[4].high
                        cor_low = waves_up[4].low
    
    print("Corrective Index at", cor_end)
    
    # imp_plot_find.write(f"##### **Corrective Index starts at {cor_end}**")
    imp_plot_find.empty()

except Exception as e:
    st.exception(e)

print_cor_idx = st.empty()
print_cor_idx.write(f"##### **Corrective Index starts at {cor_end}**")
time.sleep(2)
print_cor_idx.empty()

end_date = str(cor_date)
imp_end_date_idx = int(np.where(df_data['Date'] == end_date)[0])


# Check Corrective Plotting

try:
    with st.spinner("#### **Corrective Pattern Plotting in Progress**"):
        time.sleep(2)
        cor_plot_find = st.empty()
        wave_cycles = set()
        for new_option_correction in wave_options_correction.options_sorted:
            waves_cor = wa.find_corrective_wave(idx_start=cor_end, wave_config=new_option_correction.values) #idx_start=cor_end

            if waves_cor:
                wavepattern_cor = WavePattern(waves_cor, verbose=True)

                for rule in rules_to_check:

                    if wavepattern_cor.check_rule(rule):
                        if wavepattern_cor in wavepatterns_down:
                            continue
                        else:
                            wavepatterns_down.append(wavepattern_cor)
                            # cor_plot_find.write(f'{rule.name} found: {new_option_correction.values}')
        cor_plot_find.write(f'##### **Corrective Pattern found at: {new_option_correction.values[:3]}**')
        time.sleep(2)
    cor_plot_find.empty()

except Exception as e:
    st.exception(e)

# Plot the data using st.Plotly using st.expander

with st.expander(label="##### **View Data**"):
    data_plotting = go.Ohlc(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )
    
    data_fig = go.Figure(data=data_plotting)
    data_fig.update(layout_xaxis_rangeslider_visible=False)
    data_fig.update_layout(
        title = dict(text="Asset Data", 
                    font=dict(size=20)
                    )
    )

    st.plotly_chart(data_fig, use_container_width=True)

# Plot Impuslive Wave and show data and dataframe using st.tabs

with st.expander(label="##### **Impulsive Wave**"):
    imp_plot_tab, imp_data_tab = st.tabs([" ###### 📈 **Plotting**", " ###### 📄 **Data**"]) 

    with imp_plot_tab:
        
        try:
            plot_pattern(df=df, wave_pattern=wavepatterns_up[-1], title="Impulsive Wave")
        
        except Exception as e:
            st.exception(e)
    
    with imp_data_tab:
        hide_dataframe_row_index =  """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """
        st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
        st.dataframe(data=df.iloc[:, 0:5], use_container_width=True)

# Plot Complete Wave and Corrective Wave

with st.expander(label="##### **Prediction Waves**"):
    comp_plot_tab, cor_plot_tab = st.tabs([" ###### 💹 **Complete Cycle**", " ###### 📉 **Corrective Wave**"])

    with comp_plot_tab:
        
        try:  
            completeWave=WaveCycle(wavepatterns_up[-1],wavepatterns_down[-1])
            plot_cycle(df=df, wave_cycle=completeWave, title="Elliot Wave Plotting")

        except Exception as e:
            st.exception(e)
    
    with cor_plot_tab:
        
        try:  
            plot_pattern(df=df, wave_pattern=wavepatterns_down[-1], title="Corrective Wave")

        except Exception as e:
            st.exception(e)


# """
# Deep Learning Segment
# """

# Dataset for high and low

dataset_train_high = df_data.iloc[:, 2:3]
dataset_train_low = df_data.iloc[:, 3:4]

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))

# Dataset scaling

dataset_train_high_scaled = sc.fit_transform(dataset_train_high)
dataset_train_low_scaled = sc.fit_transform(dataset_train_low)

# """
# Data Transformation
# <--- axis 1 --->   axis 2                 axis 1 -> rows 
# X_train 60th - ith date in col 0          axis 2 -> cols
# y_train is ith date
# """

# High Data transformation

X_train_high = []
y_train_high = []
for i in range(60, len(dataset_train_high_scaled)):
    X_train_high.append(dataset_train_high_scaled[i-60:i, 0])
    y_train_high.append(dataset_train_high_scaled[i, 0])
X_train_high, y_train_high = np.array(X_train_high), np.array(y_train_high)

# Low Data transformation

X_train_low = []
y_train_low = []
for i in range(60, len(dataset_train_low_scaled)):
    X_train_low.append(dataset_train_low_scaled[i-60:i, 0])
    y_train_low.append(dataset_train_low_scaled[i, 0])
X_train_low, y_train_low = np.array(X_train_low), np.array(y_train_low)

# """
# X_train reshape process
# requires 3d data for lstm
# 1) batch_size -> np.shape[0]
# 2) timesteps or lookback -> np.shape[1]
# 3) input dimension -> 1 because time series data so 1d data 

# This reshape converts 2d -> 3d 
# """

X_train_high = np.reshape(X_train_high, (X_train_high.shape[0], X_train_high.shape[1], 1))
X_train_low = np.reshape(X_train_low, (X_train_low.shape[0], X_train_low.shape[1], 1))

# """
# Import Deep Learning Modules
# """

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Bidirectional, LSTM
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.models import save_model, load_model

# CNN Bi-LSTM Model Creation with 150 epochs

# model_create = Sequential()

# model_create = Sequential()

# # CNN

# model_create.add(Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=(X_train_open.shape[1], 1)))
# model_create.add(MaxPooling1D(pool_size=1))

# # LSTM

# model_create.add(Bidirectional(LSTM(50, return_sequences=True)))
# model_create.add(Dropout(0.2)) # Dropout 20% of neurons

# model_create.add(Bidirectional(LSTM(50, return_sequences=True)))
# model_create.add(Dropout(0.1))

# model_create.add(Bidirectional(LSTM(50, return_sequences=True)))
# model_create.add(Dropout(0.1))

# model_create.add(LSTM(50))
# model_create.add(Dropout(0.2))

# model_create.add(Flatten())

# model_create.add(Dense(1))

# model_create.compile(optimizer='adam', loss='mean_squared_error')

# model_create.name("CNN Bi-LSTM")

# Run epochs 
# model_create.fit(X_train_high, y_train_high, epochs = 150, batch_size = 32)
# model_create.save('save_model/open_model')


with st.container():

    
    with st.spinner('#### **Loading Deep Learning Models**'):
        model_open = load_model('save_model/open_model')
        model_high = load_model('save_model/high_model')
        model_high._name="CNN_Bi-LSTM"

        model_low = load_model('save_model/low_model')
        model_close = load_model('save_model/close_model')
    
    model_success = st.empty()
    model_success.success(" ##### **Model Loaded Successfully**")
    time.sleep(2)
    model_success.empty()
    with st.expander(label="##### **Model Summary**"):
        model_high.summary(print_fn=lambda x: st.text(x))



# Test Datasets

dataset_test_open = df_data.iloc[imp_end_date_idx:, 1:2]
dataset_test_high = df_data.iloc[imp_end_date_idx:, 2:3]
dataset_test_low = df_data.iloc[imp_end_date_idx:, 3:4]
dataset_test_close = df_data.iloc[imp_end_date_idx:, 4:5]

# """
# Predictions of Each Values
# """

# Open Value Prediction

dataset_total_open = pd.concat((df_data['Open'], dataset_test_open['Open']), axis = 0) # horizontal concatenation 1 vertical is 0 -> cols
inputs_open = dataset_total_open[len(dataset_total_open) - len(dataset_test_open) - 60:].values
inputs_open = inputs_open.reshape(-1,1)
inputs_open = sc.transform(inputs_open)
X_test_open = []

for i in range(60, len(inputs_open)): # for 6 months
    X_test_open.append(inputs_open[i-60:i, 0])
X_test_open = np.array(X_test_open)
X_test_open = np.reshape(X_test_open, (X_test_open.shape[0], X_test_open.shape[1], 1))
predicted_stock_price_open = model_high.predict(X_test_open)
predicted_stock_price_open = sc.inverse_transform(predicted_stock_price_open)

# High Value Prediction

dataset_total_high = pd.concat((df_data['High'], dataset_test_high['High']), axis = 0) # horizontal concatenation 1 vertical is 0 -> cols
inputs_high = dataset_total_high[len(dataset_total_high) - len(dataset_test_high) - 60:].values
inputs_high = inputs_high.reshape(-1,1)
inputs_high = sc.transform(inputs_high)
X_test_high = []

for i in range(60, len(inputs_high)): # for 6 months
    X_test_high.append(inputs_high[i-60:i, 0])
X_test_high = np.array(X_test_high)
X_test_high = np.reshape(X_test_high, (X_test_high.shape[0], X_test_high.shape[1], 1))
predicted_stock_price_high = model_high.predict(X_test_high)
predicted_stock_price_high = sc.inverse_transform(predicted_stock_price_high)

# Low Value Prediction

dataset_total_low = pd.concat((df_data['Low'], dataset_test_low['Low']), axis = 0) # horizontal concatenation 1 vertical is 0 -> cols
inputs_low = dataset_total_low[len(dataset_total_low) - len(dataset_test_low) - 60:].values
inputs_low = inputs_low.reshape(-1,1)
inputs_low = sc.transform(inputs_low)
X_test_low = []

for i in range(60, len(inputs_low)): # for 6 months
    X_test_low.append(inputs_low[i-60:i, 0])
X_test_low = np.array(X_test_low)
X_test_low = np.reshape(X_test_low, (X_test_low.shape[0], X_test_low.shape[1], 1))
predicted_stock_price_low = model_low.predict(X_test_low)
predicted_stock_price_low = sc.inverse_transform(predicted_stock_price_low)

# Close Value Prediction

dataset_total_close = pd.concat((df_data['Close'], dataset_test_close['Close']), axis = 0) # horizontal concatenation 1 vertical is 0 -> cols
inputs_close = dataset_total_close[len(dataset_total_close) - len(dataset_test_close) - 60:].values
inputs_close = inputs_close.reshape(-1,1)
inputs_close = sc.transform(inputs_close)
X_test_close = []

for i in range(60, len(inputs_close)): # for 6 months
    X_test_close.append(inputs_close[i-60:i, 0])
X_test_close = np.array(X_test_close)
X_test_close = np.reshape(X_test_close, (X_test_close.shape[0], X_test_close.shape[1], 1))
predicted_stock_price_close = model_low.predict(X_test_close)
predicted_stock_price_close = sc.inverse_transform(predicted_stock_price_close)


# Combining Predicted Values with Date

predicited_dates = df_data.iloc[imp_end_date_idx:, 0].values
df_pred = pd.DataFrame()
df_pred['Date'] = predicited_dates
df_pred['Open'] = np.array(predicted_stock_price_open)
df_pred['High'] = np.array(predicted_stock_price_high)
df_pred['Low'] = np.array(predicted_stock_price_low)
df_pred['Close'] = np.array(predicted_stock_price_close)

# Plot Real Vs Predicited data

with st.container():
    
    with st.expander("##### **Real Time Data**"):
    
        real_data_plotting = go.Ohlc(x=df_data.iloc[imp_end_date_idx:, 0],
                   open=df_data.iloc[imp_end_date_idx:, 1],
                   high=df_data.iloc[imp_end_date_idx:, 2],
                   low=df_data.iloc[imp_end_date_idx:, 3],
                   close=df_data.iloc[imp_end_date_idx:, 4])
        
        real_data_layout = dict(title="Real Time Data")
        real_data_fig = go.Figure(data=real_data_plotting)

        real_data_fig.update(layout_xaxis_rangeslider_visible=False)
        real_data_fig.update_layout(
        title = dict(text="Observed Data", 
                    font=dict(size=20)
                    )
        )
        

        st.plotly_chart(real_data_fig, use_container_width=True)
    
    with st.expander("##### **Predicted Data**"):

        pred_data_open = go.Scatter(x=df_pred['Date'].iloc[1:],
                           y=df_pred['Open'].iloc[1:],
                           name="Open Value")
        pred_data_high = go.Scatter(x=df_pred['Date'].iloc[1:],
                           y=df_pred['High'].iloc[1:],
                           name="High Value")
        pred_data_low = go.Scatter(x=df_pred['Date'].iloc[1:],
                           y=df_pred['Low'].iloc[1:],
                           name="Low Value")
        pred_data_close = go.Scatter(x=df_pred['Date'].iloc[1:],
                           y=df_pred['Close'].iloc[1:],
                           name="Close Value")
        
        
        pred_data_fig = go.Figure(data=[pred_data_open, pred_data_high, pred_data_low, pred_data_close])
        pred_data_fig.update(layout_xaxis_rangeslider_visible=False)
        pred_data_fig.update_layout(
        title = dict(text="From CNN Bi-LSTM Model", 
                    font=dict(size=20)
                    )
        )
        
        pred_plot_tab, pred_plot_data = st.tabs([" ###### 📈 **Plotting**", " ###### 📄 **Data**"])
        with pred_plot_tab:
            st.plotly_chart(pred_data_fig, use_container_width=True)
        
        with pred_plot_data:
            hide_dataframe_row_index =  """
                <style>
                .row_heading.level0 {display:none}
                .blank {display:none}
                </style>
                """
            st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
            st.dataframe(data=df_pred, use_container_width=True)


# """
# Analyse Corrective on Predicited values
# """

wa_pred = WaveAnalyzer(df=df_pred, verbose=False)
wavepatterns_down_pred = []
wave_options_correction_pred = WaveOptionsGenerator3(up_to=9)
idx_start_cor = np.argmax(np.array(list(df_pred['High'])))
try:
    wave_cycles = set()
    for new_option_correction_pred in wave_options_correction_pred.options_sorted:
        waves_cor_pred = wa_pred.find_corrective_wave(idx_start=idx_start_cor, wave_config=new_option_correction_pred.values)

        if waves_cor_pred:
            wavepattern_cor_pred = WavePattern(waves_cor_pred, verbose=True)

            for rule in rules_to_check:

                if wavepattern_cor_pred.check_rule(rule):
                    if wavepattern_cor_pred in wavepatterns_down_pred:
                        continue
                    else:
                        wavepatterns_down_pred.append(wavepattern_cor_pred)
                        # print(f'{rule.name} found: {new_option_correction_pred.values}')
    
except Exception as e:
    st.exception(e)

with st.expander(label="##### **Analysed Corrective Wave**"):
    plot_pattern_two(df=df_pred, wave_pattern=wavepattern_cor_pred, title="Updated Corrective Wave")
# print(df_pred)
