# preprocess data
import numpy as np
import modin.pandas as mpd
from util import myf
from tqdm import tqdm
import os
import ray
from sklearn.preprocessing import LabelEncoder
import warnings
from ssqueezepy import TestSignals, ssq_cwt, Wavelet, issq_cwt, cwt
import torch
import math

warnings.filterwarnings(action="ignore")

use_columns = [
    "time",
    "eq_no",
    "axis2_mac",
    "axis2_load",
    "axis2_curr",
    "axis1_mac",
    "axis1_load",
    "axis1_curr",
    "tool_num",
    "spindle_rpm",
    "spindle_load",
    "macro500",
    "auto",
    "rapid_or",
    "feedrate",
    "axisfeed",
    "count",
    "mode",
    "cycletime_min",
    "cycletime_ms",
    "spindle_temp",
    "program_m",
    "axis2_mtemp",
    "axis1_mtemp",
    "readtime",
    "axis1_abs",
    "axis2_abs",
    "axis1_dis",
    "axis2_dis",
    "program_s",
    "feedrate",
]

drop_column = [
    # "feedrate",
    "cycletime_min",
    "cycletime_ms",
    # "program_m",
    "program_s",
    # "rapid_or",
    # "spindle_temp",
    # "axis1_curr",
    # "axis2_curr",
    # "axis2_mtemp",
    # "axis1_mtemp",
]


def load_data(path, sampling_rate="50ms"):
    # utf-8인 경우
    try:
        df = mpd.read_csv(
            path, usecols=use_columns + ["axisfeed", "time"]
        )  # 특정 열만 선택하여 데이터 로드
    # euc-kr인 경우
    except:
        df = mpd.read_csv(
            path, encoding="euc-kr", usecols=use_columns + ["axisfeed", "time"]
        )  # 특정 열만 선택하여 데이터 로드
    # df = mpd.read_csv(path, usecols=use_columns + ['axisfeed', 'time']) # 특정 열만 선택하여 데이터 로드
    df = df.dropna(subset=["axisfeed"])  # 결측값 제거

    # 새로운 "readtime" 칼럼 생성 및 인덱스 설정
    df["readtime"] = df["time"].str[:10] + " " + df["readtime"]
    df.index = mpd.to_datetime(df["readtime"], format="%Y-%m-%d %H:%M:%S:%f")
    df = df.sort_index()

    df = df.drop(columns=["time", "readtime", *drop_column])  # 불필요한 열 제거
    df = df[df["count"] != 0]  # 'count'가 0인 행 제거

    le = LabelEncoder()
    cols_to_encode = df.select_dtypes(
        include="object").columns.to_list() + ["tool_num"]
    for col in cols_to_encode:
        df[col] = df[col].astype(int if col == "tool_num" else str)
        df[col] = le.fit_transform(df[col])
        # 튜플로 출력
        print(f"{col} : {dict(zip(le.classes_, range(len(le.classes_))))}")

    # 'eq_no' 별로 데이터프레임 분리
    dfs = [group for _, group in df.groupby("eq_no")]

    # 'auto'와 'mode' 열의 특정 값은 nan으로 치환
    for df in dfs:
        df.loc[df["auto"] == 3, "auto"] = np.nan
        df.loc[df["mode"] == 4, "mode"] = np.nan

    # 시간 간격을 50ms로 맞추기
    dfs_noresample = dfs
    dfs = [df.resample(sampling_rate).mean() for df in dfs]

    # 'macro500' 값이 0.5 이상인 경우 1로 치환
    for df in dfs:
        df.loc[df["macro500"] >= 0.5, "macro500"] = 1

    # 'tool_num'을 ont-hot encoding
    dfs = [
        mpd.concat([df, mpd.get_dummies(df["tool_num"].map(str))], axis=1) for df in dfs
    ]

    return dfs


def process_cwt(
    df, column, overlap=5000, length=500000, inverse=False, max_freq=220, min_freq=50
):
    num = math.ceil(len(df[column].interpolate().values) / length)

    raw_x = []  # 겹치지 않은 원래 데이터를 저장할 리스트
    x = []
    for i in range(num):
        start_idx = max(0, i * length - overlap)
        end_idx = (i + 1) * length + overlap
        x_segment = df[column].interpolate().values[start_idx:end_idx]

        # 겹치지 않는 부분만 raw_x에 추가
        if i == 0:
            raw_x.append(x_segment[:length])
        elif i == num - 1:
            raw_x.append(x_segment[overlap:])
        else:
            raw_x.append(x_segment[overlap:-overlap])

        x.append(x_segment)

    TWx_list = []
    Wx_list = []
    iwx_list = []

    for i, p in enumerate(x):
        os.environ["SSQ_GPU"] = "1"
        wavelet = Wavelet()
        TWx, Wx, *_ = ssq_cwt(p, wavelet, get_dWx=1)
        TWx = TWx.cpu().numpy()
        Wx = Wx.cpu().numpy()

        if inverse:
            os.environ["SSQ_GPU"] = "0"
            wavelet = Wavelet()
            Cs = np.repeat((max_freq + min_freq) // 2, Wx.shape[1])
            freqband = np.repeat(max_freq - min_freq, Wx.shape[1])
            iwx = issq_cwt(Wx, wavelet, Cs, freqband)[0]
            iwx_list.append(iwx)

        # TWx = TWx
        # Wx = Wx
        # 겹치는 부분 제거
        if i > 0:
            Wx = Wx[:, overlap:]
            TWx = TWx[:, overlap:]
        if i < num - 1:
            Wx = Wx[:, :-overlap]
            TWx = TWx[:, :-overlap]

        TWx_list.append(TWx)
        Wx_list.append(Wx)
        torch.cuda.empty_cache()

    Wx = np.concatenate([t for t in Wx_list], axis=1)
    TWx = np.concatenate([t for t in TWx_list], axis=1)
    raw_x = np.concatenate(raw_x)  # 겹치지 않는 원래 형태의 데이터
    Iwx = np.concatenate([t for t in iwx_list], axis=0)

    return Wx, TWx, raw_x, Iwx


def process_cwt_only(
    df, column, overlap=5000, length=500000, inverse=False, max_freq=220, min_freq=50
):
    num = math.ceil(len(df[column].interpolate().values) / length)

    raw_x = []  # 겹치지 않은 원래 데이터를 저장할 리스트
    x = []
    for i in range(num):
        start_idx = max(0, i * length - overlap)
        end_idx = (i + 1) * length + overlap
        x_segment = df[column].interpolate().values[start_idx:end_idx]

        # 겹치지 않는 부분만 raw_x에 추가
        if i == 0:
            raw_x.append(x_segment[:length])
        elif i == num - 1:
            raw_x.append(x_segment[overlap:])
        else:
            raw_x.append(x_segment[overlap:-overlap])

        x.append(x_segment)

    Wx_list = []
    iwx_list = []

    for i, p in enumerate(x):
        os.environ["SSQ_GPU"] = "1"
        wavelet = Wavelet()
        Wx, _ = cwt(p, wavelet)
        Wx = Wx.cpu().numpy()

        if inverse:
            os.environ["SSQ_GPU"] = "0"
            wavelet = Wavelet()
            Cs = np.repeat((max_freq + min_freq) // 2, Wx.shape[1])
            freqband = np.repeat(max_freq - min_freq, Wx.shape[1])
            iwx = issq_cwt(Wx, wavelet, Cs, freqband)[0]
            iwx_list.append(iwx)

        # 겹치는 부분 제거
        if i > 0:
            Wx = Wx[:, overlap:]
        if i < num - 1:
            Wx = Wx[:, :-overlap]

        Wx_list.append(Wx)
        torch.cuda.empty_cache()

    Wx = np.concatenate([t for t in Wx_list], axis=1)
    raw_x = np.concatenate(raw_x)  # 겹치지 않는 원래 형태의 데이터

    if inverse:
        Iwx = np.concatenate([t for t in iwx_list], axis=0)
        return Wx, raw_x, Iwx
    else:
        return Wx, raw_x
