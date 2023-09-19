from util import myf
from util import preprocess
from ssqueezepy import TestSignals, ssq_cwt, Wavelet, issq_cwt, cwt
import matplotlib.pyplot as plt
import cv2
from IPython.display import clear_output
import os
from functools import reduce
import operator
import warnings
from ipywidgets.widgets import *
from ipywidgets import widgets, Layout
from plotly_resampler import FigureWidgetResampler
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
import stumpy
import plotly.graph_objects as go
import sys

sys.path.append("/mnt/c/lab/연구/공구/util")
sys.path.append("/rapids/notebooks/lab/연구/공구/util/")

warnings.filterwarnings(action="ignore")


class CwtMatcher:
    def __init__(self, df_list, label):
        self.ts_figure = go.FigureWidget()
        self.df_list = df_list

        self.label = label

        self.df_list_dropdown = Dropdown(
            options=np.arange(len(df_list)), description="df_list"
        )
        self.sampling_rate_text = Text(
            value="1000ms", description="sampling_rate")
        self.button1 = Button(description="확인")
        self.button1.on_click(self.create_rule)
        self.progress_bar = IntProgress(
            min=0, max=100, description="Progress:")

        self.eq_list_dropdown = Dropdown(
            options=["eq1", "eq2", "eq3"], description="eq_list"
        )
        self.rule_dropdown = Dropdown(description="rule_list")
        self.button2 = Button(description="시각화")
        self.button2.on_click(self.create_plot)
        self.button2_box = HBox([])

        self.plot_box = VBox([])
        self.plot_progress_bar = IntProgress(
            min=0, max=100, description="Progress:")

        self.cwt_dropdown = Dropdown(description="cwt_column")
        self.cwt_button = Button(description="cwt 변환")
        self.icwt_button = Button(description="icwt 변환")
        self.cwt_button.on_click(self.transform_cwt)
        self.icwt_button.on_click(self.transform_icwt)
        self.cwt_image = Output()
        self.icwt_plot = HBox([])
        self.icwt_box = HBox([])
        self.mp_box = HBox([])
        self.mp_box2 = HBox([])

        self.wd = VBox(
            [
                HBox([self.df_list_dropdown, self.sampling_rate_text, self.button1]),
                HBox([self.progress_bar]),
                HBox(
                    [
                        self.eq_list_dropdown,
                        self.rule_dropdown,
                        self.button2_box,
                    ]
                ),
                self.plot_progress_bar,
                self.plot_box,
                self.cwt_image,
                HBox([self.cwt_dropdown, self.cwt_button, self.icwt_button]),
                self.icwt_box,
                self.icwt_plot,
                self.mp_box,
                self.mp_box2
            ],
        )

    def create_rule(self, clicked_button: widgets.Button):
        self.progress_bar.value = 10
        df = self.df_list[self.df_list_dropdown.value]
        df = preprocess.load_data(df, self.sampling_rate_text.value)
        self.progress_bar.value = 50
        self.df = df

        rules_eq1_tool1 = [
            df[0]["axis1_mac"] < -626,
            df[0]["axis1_mac"] > -654,
            df[0]["axis1_abs"] > -1.19,
        ]

        # dfs[0] 설비1 tool 2의 규칙
        rules_eq1_tool2 = [
            df[0]["axis2_mac"] < -280,
            df[0]["axis2_mac"] > -294.61,
            df[0]["axis1_mac"] > -494,
            df[0]["axis1_mac"] < -477,
            df[0]["axis2_dis"] < 10,
        ]

        rules_eq2_tool1 = [
            df[1]["axis1_mac"] < -605.151,
            df[1]["axis1_mac"] > -633.798,
            df[1]["axis1_abs"] >= 24,
            df[1]["axis1_abs"] <= 61,
        ]

        rules_eq3_tool1 = [
            # dfs[2]['axis2_mac'] < -234,
            # dfs[2]['axis2_mac'] > -246,
            df[2]["axis1_mac"] < -500,
            df[2]["axisfeed"] == 184,
        ]

        rules_eq3_tool2 = [
            df[2]["axis2_mac"] < -231,
            df[2]["axis2_mac"] > -246,
            df[2]["axis1_mac"] > -200,
        ]
        # dict로 저장
        self.rules = {
            "eq1_tool1": rules_eq1_tool1,
            "eq1_tool2": rules_eq1_tool2,
            "eq2_tool1": rules_eq2_tool1,
            "eq3_tool1": rules_eq3_tool1,
            "eq3_tool2": rules_eq3_tool2,
        }

        self.rule_dropdown.options = list(self.rules.keys())
        self.button2_box.children = [self.button2]
        self.cwt_dropdown.options = list(df[0].columns)

        self.progress_bar.value = 100

    def create_plot(self, clicked_button: widgets.Button):
        self.plot_progress_bar.value = 10
        if self.eq_list_dropdown.value == "eq1":
            select = 0
        elif self.eq_list_dropdown.value == "eq2":
            select = 1
        elif self.eq_list_dropdown.value == "eq3":
            select = 2

        df_tmp = self.df[select]
        df_tmp = df_tmp.interpolate()[
            reduce(operator.and_, self.rules[self.rule_dropdown.value])
        ]
        self.select_df_index = df_tmp
        self.select_df = df_tmp.reset_index(drop=True)

        fig = myf.visualing(self.select_df, scaler="minmax",
                            width=1000, verbose=False)

        self.plot_box.children = [fig]
        self.plot_progress_bar.value = 100

    def transform_cwt(self, clicked_button: widgets.Button):
        self.cwt_image.clear_output(wait=True)
        # self.select_df[column]을 cwt 변환
        wx, _ = preprocess.process_cwt_only(
            self.select_df, self.cwt_dropdown.value, 5000, 500000, inverse=False
        )
        with self.cwt_image:
            wx_abs = np.abs(wx).astype(np.float32)
            wx_resize = cv2.resize(wx_abs, (0, 0), fx=0.1, fy=1)
            plt.figure(figsize=(13, 5))
            plt.imshow(wx_resize, cmap="turbo", aspect="auto")
            plt.title(self.cwt_dropdown.value)
            plt.show()

        self.slider = IntRangeSlider(
            value=[0, wx.shape[0]],
            min=0,
            max=wx.shape[0],
            step=1,
            description="BPF:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            layout=Layout(width="500px"),
        )
        self.icwt_progress_bar = IntProgress(
            min=0, max=100, description="Progress:")
        self.icwt_box.children = [self.slider, self.icwt_progress_bar]
        self.wx = wx

    def transform_icwt(self, clicked_button: widgets.Button):
        # self.wx를 icwt 변환
        self.icwt_progress_bar.value = 10

        os.environ["SSQ_GPU"] = "0"
        wavelet = Wavelet()

        cs = np.repeat(
            (self.slider.value[0] + self.slider.value[1]) // 2,
            self.wx.shape[1],
        )
        freqband = np.repeat(
            (self.slider.value[1] - self.slider.value[0]) // 2,
            self.wx.shape[1],
        )
        self.iwx = pd.DataFrame(issq_cwt(self.wx, wavelet, cs, freqband)[
                                0], index=self.select_df_index.index)

        date = self.select_df_index.index.strftime("%Y-%m-%d")[0]
        label = self.label.loc[date, self.rule_dropdown.value]
        label = label[label == 1]

        fig = go.FigureWidget(make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02))

        fig.append_trace(
            go.Scattergl(
                x=self.iwx.index,
                y=np.abs(self.wx)[self.slider.value[0]
                         :self.slider.value[1]].mean(axis=0),
            ), row=1, col=1
        )

        fig.append_trace(
            go.Scattergl(
                x=self.iwx.index,
                y=self.iwx.values.flatten(),
            ), row=2, col=1
        )

        for idx in label.index:
            fig.add_vline(x=idx, line_width=3,
                          line_color="green")

        fig.update_layout(height=500, width=1000)
        self.icwt_plot.children = [fig]

        self.cwt_image.clear_output(wait=True)
        with self.cwt_image:
            wx_resize = cv2.resize(np.abs(self.wx), (0, 0), fx=0.1, fy=1)
            plt.figure(figsize=(13, 5))
            plt.imshow(wx_resize, cmap="turbo", aspect="auto")
            plt.title(self.cwt_dropdown.value)
            plt.axhline(self.slider.value[0], color="r", linewidth=1)
            plt.axhline(self.slider.value[1], color="r", linewidth=1)
            plt.show()

        self.mp_button = widgets.Button(description="matirx profile")
        self.mp_button.on_click(self.transform_mp)
        self.mp_length = widgets.IntText(
            value=100,
            description="length:",
        )
        self.mp_box.children = [self.mp_button, self.mp_length]
        self.icwt_progress_bar.value = 100

    def transform_mp(self, clicked_button: widgets.Button):
        iwx = self.iwx
        self.icwt_progress_bar.value = 10
        mp = stumpy.gpu_stump(iwx.values.flatten(),
                              self.mp_length.value, normalize=False)[:, 0]

        self.icwt_progress_bar.value = 30
        mp2 = stumpy.gpu_stump(
            self.select_df[self.cwt_dropdown.value].values, self.mp_length.value, normalize=False)[:, 0]

        date = self.select_df_index.index.strftime("%Y-%m-%d")[0]
        label = self.label.loc[date, self.rule_dropdown.value]
        label = label[label == 1]

        fig = go.FigureWidget(make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=(
            "cwt, bpf, mean", "iwct, bpf", "ts, matrix profile", "iwct, matrix profile")))

        self.icwt_progress_bar.value = 50
        fig.append_trace(
            go.Scattergl(
                x=self.iwx.index,
                y=np.abs(self.wx)[self.slider.value[0]                                  :self.slider.value[1]].mean(axis=0),
            ), row=1, col=1
        )

        fig.append_trace(
            go.Scattergl(
                x=self.iwx.index,
                y=self.iwx.values.flatten(),
            ), row=2, col=1
        )
        fig.append_trace(
            go.Scattergl(
                x=self.iwx.index[self.mp_length.value:],
                y=mp2,
            ), row=3, col=1
        )
        fig.append_trace(
            go.Scattergl(
                x=self.iwx.index[self.mp_length.value:],
                y=mp,
            ), row=4, col=1
        )

        for idx in label.index:
            fig.add_vline(x=idx, line_width=3,
                          line_color="green")

        fig.update_layout(height=1000, width=1000)
        self.icwt_plot.children = [fig]
        self.icwt_progress_bar.value = 100

    def display(self):
        return self.wd
