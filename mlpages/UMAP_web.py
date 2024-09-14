import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import cm
import argparse
from sklearn.decomposition import PCA
import os
import zipfile
import warnings
import streamlit as st

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

def plot_point_cov(points, nstd=3, ax=None, **kwargs):
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=3, ax=None, **kwargs):
    def eigsorted(cov):
        cov = np.array(cov)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]
    if ax is None:
        ax = plt.gca()
    vals, vecs = eigsorted(cov)

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_artist(ellip)
    return ellip

'''画置信圆'''
def show_ellipse(X, y, prefix, x1, x2, y1, y2, output):
    model = PCA(n_components=2)
    embedding = model.fit_transform(X)

    unique_drugs = set(y)
    regions = sorted(list(unique_drugs))

    labels_prefix = [f"{prefix}{region}" for region in regions]

    color_map = plt.get_cmap('rainbow')
    color_indices = np.linspace(0, 1, len(regions))

    # 定义分辨率
    plt.figure(dpi=300, figsize=(3.5, 3))
    # 三分类则为3
    for i in range(0, len(regions), 1):
        colors = color_map(color_indices[i])
        pts = embedding[y == int(i+1), :]
        new_x, new_y = embedding[y==i+1, 0], embedding[y==i+1, 1]
        plt.plot(new_x, new_y, 'o', color=colors, label=labels_prefix[i], markersize=3)
        plot_point_cov(pts, nstd=3, alpha=0.25, color=colors)

    # 添加坐标轴
    plt.xlim(np.min(new_x)-x1, np.max(new_x)+x2)
    plt.ylim(np.min(new_y)-y1, np.max(new_y)+y2)
    plt.xticks(size=5, color='black')
    plt.yticks(size=5, color='black')
    plt.xlabel('PC1 ({} %)'.format(round(model.explained_variance_ratio_[0] * 100, 2)), fontsize=6, color='black')
    plt.ylabel('PC2 ({} %)'.format(round(model.explained_variance_ratio_[1] * 100, 2)), fontsize=6, color='black')
    plt.legend(prop={"size": 5},  loc='upper right', frameon=False, edgecolor='none', facecolor='none')
    # plt.savefig(f'{output}_plot1.png', bbox_inches='tight', dpi=300)
    # plt.savefig(f'{output}_plot1.eps', bbox_inches='tight', dpi=300)
    # plt.show()
    plt.title(f'{output}', size=7, color='black')
    st.pyplot(plt.gcf())


def show_ratio(X, output):
    st.subheader("Variance Ratio")
    model = PCA()
    embedding = model.fit_transform(X)

    cumulative_variance = []
    cumulative_sum = 0
    for ratio in list(model.explained_variance_ratio_):
        cumulative_sum += ratio*100
        cumulative_variance.append(cumulative_sum)

    PCA_var = pd.DataFrame(cumulative_variance)
    # PCA_var.to_csv(f"{output}_var.csv")
    st.write(PCA_var)


    plt.figure(dpi=300, figsize=(3.5, 3))
    plt.bar(range(len(cumulative_variance)), cumulative_variance, color = '#2d857a', width=0.35)
    for i in range(len(cumulative_variance)):
        plt.text(x=i, y=cumulative_variance[i]+0.5, s=str(round(cumulative_variance[i], 2)), ha='center', va='bottom', fontsize=5, color='black')
    plt.xticks(size=5, color='black')
    plt.yticks(size=5, color='black')
    plt.xlabel('Number of PCs', fontsize=6, color='black')
    plt.ylabel('Variance Ratio (%)', fontsize=6, color='black')
    # plt.savefig(f'{output}_plot2.png', bbox_inches='tight', dpi=300)
    # plt.savefig(f'{output}_plot2.eps', bbox_inches='tight', dpi=300)
    # plt.show()
    plt.title(f'{output}', size=7, color='black')
    st.pyplot(plt.gcf())


# def parameter():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-f', type=str, default='data.csv', help='csv数据文件')
#     parser.add_argument('-n', type=str, default='name', help='样本名称的列名')
#     parser.add_argument('-g', type=str, default='group', help='样本分组的列名')
#     parser.add_argument('-x1', type=float, default=None, help='x轴上限')
#     parser.add_argument('-x2', type=float, default=None, help='x轴下限')
#     parser.add_argument('-y1', type=float, default=None, help='y轴上限')
#     parser.add_argument('-y2', type=float, default=None, help='y轴下限')
#     parser.add_argument('-o', type=str, default='PCAoutput', help='默认输出前缀')
#     args = parser.parse_args()
#     return args


@st.cache_resource
def load_data(Inputfile):
    return pd.read_csv(Inputfile)


if __name__ == '__main__':
    # args = parameter()

    # data = pd.read_csv(args.f)
    # X = data.drop([args.n, args.g], axis=1)
    # labels, y = data[args.g], data[args.g]

    # show_ellipse(X, y, args.g, args.x1, args.x2, args.y1, args.y2, args.o)
    # show_ratio(X, args.o)

    uploaded_file = st.sidebar.file_uploader("请上传csv数据文件")

    if st.sidebar.checkbox("加载案例数据", False):
        uploaded_file = './data.csv'

    if uploaded_file is not None:
        data = load_data(uploaded_file)

    if st.sidebar.checkbox("Show dataset", True):
        st.subheader("DataSet")
        st.write(df)

    name = st.sidebar.text_input("样本名称的列名")
    group = st.sidebar.text_input("样本分组的列名")

    X = data.drop([name, group], axis=1)
    y = data[group]

    x1 = st.sidebar.text_input("x轴上限-")
    x2 = st.sidebar.text_input("x轴上限+")
    y1 = st.sidebar.text_input("y轴上限-")
    y2 = st.sidebar.text_input("y轴上限+")

    Output = st.sidebar.text_input("Enter label name 👇")


    show_ellipse(X, y, group, x1, x2, y1, y2, Output)
    show_ratio(X, Output)
