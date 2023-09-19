import streamlit as st
import pandas as pd
from util import myf
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

dfs = []
for i in range(3):
    df = pd.read_csv('data/dfs_{}.0.csv'.format(i))
    df.index = df['Unnamed: 0']
    df.drop('Unnamed: 0', axis=1, inplace=True)
    # scaling
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(
        df), columns=df.columns, index=df.index)
    dfs.append(df)

st.title('설비1')
st.plotly_chart(px.line(dfs[0]))
st.title('설비2')
st.plotly_chart(px.line(dfs[1]))
st.title('설비3')
st.plotly_chart(px.line(dfs[2]))
