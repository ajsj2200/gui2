import socket
import plotly.io as pio
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from util import visualizing
from multiprocessing import Process
import streamlit.components.v1 as components
import numpy as np
st.set_page_config(layout="wide")
pio.templates.default = "plotly"


def get_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


dfs = []
for i in range(3):
    df = pd.read_csv('data/dfs_{}.0.csv'.format(i))
    df.index = df['Unnamed: 0']
    df.drop('Unnamed: 0', axis=1, inplace=True)
    # scaling
    df.index = pd.to_datetime(df.index)
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(
        df), columns=df.columns, index=df.index)
    # 모두 float64로 변환
    df = df.astype('float64')
    # df.dropna(axis=0, inplace=True)
    dfs.append(df)

for idx, df in enumerate(dfs):

    fig = visualizing.visualing(
        df, scaler='minmax', show=False, width=800)

    st.title('설비{}'.format(idx+1))

    port = get_available_port()
    proc = Process(
        target=fig.show_dash, kwargs=dict(mode="external", port=port)
    ).start()
    components.iframe(f"http://localhost:{port}", height=700)

# always wide mode
