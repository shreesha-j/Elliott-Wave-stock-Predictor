from models.WavePattern import WavePattern
import pandas as pd
import time
import plotly.graph_objects as go
import streamlit as st


def timeit(func):
    def wrapper(*arg, **kw):

        t1 = time.perf_counter_ns()
        res = func(*arg, **kw)
        t2 = time.perf_counter_ns()
        print("took:", t2-t1, 'ns')
        return res
    return wrapper


def plot_cycle(df, wave_cycle, title: str = ''):

    data = go.Ohlc(x=df['Date'],
                   open=df['Open'],
                   high=df['High'],
                   low=df['Low'],
                   close=df['Close'],
                   name="OHLC")

    monowaves = go.Scatter(x=wave_cycle.dates,
                           y=wave_cycle.values,
                           text=wave_cycle.labels,
                           mode='lines+markers+text',
                           textposition='middle right',
                           textfont=dict(size=15, color='#2c3035'),
                           name="Wave Plotter",
                           line=dict(
                               color=('rgb(111, 126, 130)'),
                               width=3),
                           )
    fig = go.Figure(data=[data, monowaves])
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(
        title = dict(text=title, 
                    font=dict(size=20)
                    )
    )

    st.plotly_chart(fig, use_container_width=True)


def convert_yf_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a yahoo finance OHLC DataFrame to column name(s) used in this project

    old_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    new_names = ['Date', 'Open', 'High', 'Low', 'Close']

    :param df:
    :return:
    """
    df_output = pd.DataFrame()

    df_output['Date'] = list(df.index)
    df_output['Date'] = pd.to_datetime(df_output['Date'], format="%Y-%m-%d %H:%M:%S")

    df_output['Open'] = df['Open'].to_list()
    df_output['High'] = df['High'].to_list()
    df_output['Low'] = df['Low'].to_list()
    df_output['Close'] = df['Close'].to_list()


    return df_output

def plot_pattern(df: pd.DataFrame, wave_pattern: WavePattern, title: str = ''):
    data = go.Ohlc(x=df['Date'],
                   open=df['Open'],
                   high=df['High'],
                   low=df['Low'],
                   close=df['Close'],
                   name="OHLC")
    monowaves = go.Scatter(x=wave_pattern.dates,
                           y=wave_pattern.values,
                           text=wave_pattern.labels,
                           mode='lines+markers+text',
                           textposition='middle right',
                           textfont=dict(size=15, color='#2c3035'),
                           name="Wave Plotter",
                           line=dict(
                               color=('rgb(111, 126, 130)'),
                               width=3),
                           )
    fig = go.Figure(data=[data, monowaves])
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(
        title = dict(text=title, 
                    font=dict(size=20)
                    )
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_pattern_two(df: pd.DataFrame, wave_pattern: WavePattern, title: str = ''):
    data_high = go.Scatter(x=df['Date'].iloc[1:],
                           y=df['High'].iloc[1:],
                           name="High Value")
    data_low = go.Scatter(x=df['Date'].iloc[1:],
                           y=df['Low'].iloc[1:],
                           name="Low Value")
    monowaves = go.Scatter(x=wave_pattern.dates,
                           y=wave_pattern.values,
                           text=wave_pattern.labels,
                           mode='lines+markers+text',
                           textposition='middle right',
                           textfont=dict(size=15, color='#2c3035'),
                           line=dict(
                               color=('rgb(111, 126, 130)'),
                               width=3),
                           )
    fig = go.Figure(data=[data_high, data_low, monowaves])
    
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(
        title = dict(text=title, 
                    font=dict(size=20)
                    )
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_monowave(df, monowave, title: str = ''):
    data = go.Ohlc(x=df['Date'],
                   open=df['Open'],
                   high=df['High'],
                   low=df['Low'],
                   close=df['Close'])

    monowaves = go.Scatter(x=monowave.dates,
                           y=monowave.points,
                           mode='lines+markers+text',
                           textposition='middle right',
                           textfont=dict(size=15, color='#2c3035'),
                           line=dict(
                               color=('rgb(111, 126, 130)'),
                               width=3),
                           )
    layout = dict(title=title)
    fig = go.Figure(data=[data, monowaves], layout=layout)
    fig.update(layout_xaxis_rangeslider_visible=False)

    fig.show()
