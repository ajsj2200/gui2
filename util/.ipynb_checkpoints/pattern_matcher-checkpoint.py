import numpy as np
import pandas as pd
import stumpy
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import cuml
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from ipywidgets import widgets
import tsfresh
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from plotly_resampler import FigureWidgetResampler 
import sys
sys.path.append('/mnt/c/lab/연구/공구/util')
import myf

class MassMatcher:
    def __init__(self, df):
        self.df = df
        self.f = myf.visualing(df, width=1000, height=500,
                               scaler='minmax', show=False)
        self.f2 = FigureWidgetResampler(
            go.FigureWidget(), default_n_shown_samples=5000)
        self.f3 = go.FigureWidget()
        self.f3_single = go.FigureWidget(layout={'height':300})
        self.f4 = go.FigureWidget()
        self.f5 = go.FigureWidget()
        self.f6 = FigureWidgetResampler(
            go.FigureWidget(), default_n_shown_samples=5000)
        self.f7 = FigureWidgetResampler(
            go.FigureWidget(), default_n_shown_samples=5000)

        self.peaks = None
        self.query_length = None
        self.create_widgets()
        self.embeddings = None
        self.patterns = None
        self.select_index = None
        self.centroid_index = None

    def create_widgets(self):
        self.button = widgets.Button(description="Matching")
        self.output = widgets.Output(layout={'border': '1px solid black'})
        self.column_dropdown = widgets.Dropdown(
            options=self.df.columns, value=self.df.columns[0], description='column')
        self.t = widgets.Text(value='10000', disabled=False)
        self.distance = widgets.Text(value='500', disabled=False)
        self.button2 = widgets.Button(description="Find peaks")
        self.button3 = widgets.Button(description="Display umap")
        self.button4 = widgets.Button(description="show AVG pattern")
        self.button_pattern_features = widgets.Button(
            description="show pattern features")
        self.text_len_peak = widgets.Text(value='0', disabled=True)
        self.metric_dropdown = widgets.Dropdown(
            options=['extract feature', 'no extract feature'], value='no extract feature', description='metric')
        self.pattern_metric_dropdown = widgets.Dropdown(
            options=['mean', 'max', 'min', 'std', 'skew', 'kurt'], value='mean', description='pattern metric')
        self.button.on_click(self.calculate_mass_distance)
        self.button2.on_click(self.find_peaks_func)
        self.button3.on_click(self.display_umap)
        self.button4.on_click(self.find_closest_pattern)
        self.button_pattern_features.on_click(self.display_pattern_features)

    def calculate_mass_distance(self, clicked_button: widgets.Button):
        x_ = self.f.layout.xaxis1.range
        c = self.column_dropdown.value
        query = self.df.loc[x_[0]:x_[1], c].values

        distance = -stumpy.mass(query, self.df[c], normalize=False)

        with self.f2.batch_update():
            self.f2.data = []
            self.f2.add_trace(go.Scattergl(x=self.df.index[len(self.df.loc[x_[0]:x_[1], c])-1:], y=-distance, name='distance'),
                              hf_x=self.df.index[len(self.df.loc[x_[0]:x_[1], c])-1:], hf_y=-distance)

    def find_peaks_func(self, clicked_button: widgets.Button):
        x_ = self.f.layout.xaxis1.range
        c = self.column_dropdown.value
        query = self.df.loc[x_[0]:x_[1], c].values
        self.query_length = len(query)

        distance = -stumpy.mass(query, self.df[c], normalize=False)

        peaks, _ = find_peaks(
            distance, height=-float(self.t.value), distance=float(self.distance.value))
        self.peaks = peaks
        self.text_len_peak.value = str(len(peaks))

        with self.f2.batch_update():
            self.f2.data = []
            self.f2.add_trace(go.Scattergl(x=self.df.index[len(self.df.loc[x_[0]:x_[1], c])-1:], y=-distance, name='distance'),
                              hf_x=self.df.index[len(self.df.loc[x_[0]:x_[1], c])-1:], hf_y=-distance)
            self.f2.add_trace(go.Scattergl(x=self.df.index[len(self.df.loc[x_[0]:x_[1], c])-1:][peaks], y=-distance[peaks], mode='markers', name='peaks'),
                              hf_x=self.df.index[len(self.df.loc[x_[0]:x_[1], c])-1:][peaks], hf_y=-distance[peaks])
            self.f2.add_trace(go.Scattergl(x=self.df.index[len(self.df.loc[x_[0]:x_[1], c])-1:][peaks], y=-distance[peaks], mode='lines', name='peaks'),
                              hf_x=self.df.index[len(self.df.loc[x_[0]:x_[1], c])-1:][peaks], hf_y=-distance[peaks])

    def extract_patterns(self, column, length):
        patterns = []
        tmp = self.df[column].values
        for i in range(len(self.peaks)-1):
            patterns.append(
                tmp[self.peaks[i]:self.peaks[i]+length])
        return np.array(patterns)

    def display_umap(self, clicked_button: widgets.Button):
        patterns = self.extract_patterns(
            self.column_dropdown.value, self.query_length)

        if self.metric_dropdown.value == 'extract feature':
            umap = cuml.UMAP(verbose=False, n_epochs=500,
                             min_dist=0, n_neighbors=15)
            
            conv_df = convert_to_tsfresh_format(patterns)
            settings = ComprehensiveFCParameters()
            features = tsfresh.extract_features(conv_df, column_id='id', column_sort='time',
                                    column_value='F_x', default_fc_parameters=settings)
            features = features.dropna(axis=1).values
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
            
            embedding = umap.fit_transform(features)
            

            umap_single = cuml.UMAP(verbose=False, n_epochs=500, min_dist=0, n_neighbors=15, n_components=1)
            embedding_single = umap_single.fit_transform(features)

        elif self.metric_dropdown.value == 'no extract feature':
            umap = cuml.UMAP(verbose=False, n_epochs=500,
                             min_dist=0, n_neighbors=15)
            embedding = umap.fit_transform(patterns)
            umap_single = cuml.UMAP(verbose=False, n_epochs=500, min_dist=0, n_neighbors=15, n_components=1)
            embedding_single = umap_single.fit_transform(patterns)

        self.embeddings = embedding
        self.patterns = patterns

        # hover할 때 각 점의 시간을 보여주도록.
        hovertext = []
        for i in range(len(self.peaks)-1):
            hovertext.append(
                self.df.index[self.peaks[i]])

        with self.f3.batch_update():
            self.f3.data = []
            self.f3.add_trace(go.Scattergl(
                x=embedding[:, 0], y=embedding[:, 1], mode='markers', name='patterns', text=hovertext))
            self.f3.data[0].on_click(self.update_umap_point)

        # 시간에 따라 embedding_single을 그리기
        with self.f3_single.batch_update():
            self.f3_single.data = []
            self.f3_single.add_trace(go.Scattergl(
                x=self.df.index[self.peaks], y=embedding_single[:, 0], mode='lines', name='patterns'))
            
    def update_umap_point(self, trace, points, selector):
        # f3에서 선택한 점의 색이 바뀌도록.
        with self.f3.batch_update():
            index = points.point_inds[0]
            self.select_index = index
            self.f3.data[0].marker.color = ['red' if i ==
                                            index else 'blue' for i in range(len(self.patterns))]
            self.f3.data[0].marker.size = [10 if i ==
                                           index else 5 for i in range(len(self.patterns))]

        # f4에 선택한 점의 패턴을 그래프로 그려줌.
        with self.f4.batch_update():
            self.f4.data = []
            self.f4.add_trace(go.Scattergl(x=self.df.index[self.peaks[index]:self.peaks[index]+self.query_length],
                                           y=self.df[self.column_dropdown.value].iloc[self.peaks[index]:self.peaks[index]+self.query_length], mode='lines', name='pattern'),
                              )
            self.f4.layout.yaxis1.range = [
                self.patterns.min(), self.patterns.max()]

    # f5에 f3에 표시돼있는 점들의 평균과 가장 가까운 점을 그래프로 그려줌.
    def find_closest_pattern(self, clicked_button: widgets.Button):
        # Get current x and y range from f5
        x_range = self.f3.layout.xaxis1.range
        y_range = self.f3.layout.yaxis1.range

        # Select points that are currently displayed
        displayed_embeddings = self.embeddings[(self.embeddings[:, 0] >= x_range[0]) & (self.embeddings[:, 0] <= x_range[1]) &
                                               (self.embeddings[:, 1] >= y_range[0]) & (self.embeddings[:, 1] <= y_range[1])]

        if len(displayed_embeddings) == 0:
            print("No points are currently displayed in the selected range.")
            return

        # Calculate the centroid of displayed points
        centroid = np.mean(displayed_embeddings, axis=0)

        # Calculate the distance from the centroid to each displayed point
        dist = np.linalg.norm(displayed_embeddings - centroid, axis=1)

        # Find the index of the closest point among the displayed points
        closest_index_displayed = np.argmin(dist)

        # Find the closest point in the original embeddings
        closest_point = displayed_embeddings[closest_index_displayed]
        index = np.where((self.embeddings == closest_point).all(axis=1))[0][0]
        self.centroid_index = index

        with self.f5.batch_update():
            self.f5.data = []
            self.f5.add_trace(go.Scattergl(x=self.df.index[self.peaks[index]:self.peaks[index]+self.query_length],
                                           y=self.df[self.column_dropdown.value].iloc[self.peaks[index]:self.peaks[index]+self.query_length],
                                           mode='lines', name='pattern'),
                              )
            # y축 최대 최소값을 찾은 패턴의 최대 최소값으로 설정
            self.f5.layout.yaxis1.range = [self.patterns.min(
            ), self.patterns.max()]
        with self.f3.batch_update():
            self.f3.data[0].marker.color = ['red' if i == self.select_index else 'green' if i ==
                                            self.centroid_index else 'blue' for i in range(len(self.patterns))]
            self.f3.data[0].marker.size = [10 if i ==
                                           index else 5 for i in range(len(self.patterns))]

        query = self.df[self.column_dropdown.value].iloc[self.peaks[index]:self.peaks[index]+self.query_length].values
        distance = - \
            stumpy.mass(
                query, self.df[self.column_dropdown.value].values, normalize=False)

        # draw distance
        with self.f6.batch_update():
            self.f6.data = []
            self.f6.add_trace(go.Scattergl(
                x=self.df.index[:-len(query)+1], y=-distance, mode='lines', name='distance'))

    def display_pattern_features(self, clicked_button: widgets.Button):
        # pattern metric dropdown에 선택한 것으로 패턴 특징을 그래프로 그려줌.
        if self.pattern_metric_dropdown.value == 'mean':
            pattern_features = self.patterns.mean(axis=1)
        elif self.pattern_metric_dropdown.value == 'std':
            pattern_features = self.patterns.std(axis=1)
        elif self.pattern_metric_dropdown.value == 'max':
            pattern_features = self.patterns.max(axis=1)
        elif self.pattern_metric_dropdown.value == 'min':
            pattern_features = self.patterns.min(axis=1)
        elif self.pattern_metric_dropdown.value == 'skew':
            pattern_features = pd.DataFrame(self.patterns.T).skew().values
        elif self.pattern_metric_dropdown.value == 'kurt':
            pattern_features = pd.DataFrame(self.patterns.T).kurt().values

        with self.f7.batch_update():
            self.f7.data = []
            self.f7.add_trace(go.Scattergl(x=np.arange(len(pattern_features)), y=pattern_features, mode='lines', name='pattern features'),
                               hf_x=np.arange(len(pattern_features)), hf_y=pattern_features)

    def display(self):
        return widgets.VBox([widgets.HBox([self.f, self.f2]),
                             widgets.HBox([self.column_dropdown, self.button, self.t,
                                          self.distance, self.button2, self.text_len_peak, self.button3, self.button4, self.metric_dropdown]),
                             widgets.HBox([widgets.VBox([self.f3, self.f3_single]), self.f4]), widgets.HBox([self.f5, self.f6]),
                               widgets.VBox([self.pattern_metric_dropdown, 
                                             self.button_pattern_features, self.f7])])
    
# 이 코드는 (batch, length) 형태의 numpy array를 입력으로 받습니다.
def convert_to_tsfresh_format(data):
    ids = np.repeat(np.arange(len(data)), len(data[0]))
    times = np.tile(np.arange(len(data[0])), len(data))
    values = data.flatten()

    df = pd.DataFrame({'id': ids, 'time': times, 'F_x': values})

    return df