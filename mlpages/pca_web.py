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
import plotly.express as px

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Arial'] + plt.rcParams['font.serif'] # Times New Roman

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

def plot_point_cov(points, nstd=3, ax=None, **kwargs):
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

# 1画散点和置信圆图
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
    xxx, yyy = [], []
    for i in range(0, len(regions), 1):
        colors = color_map(color_indices[i])
        pts = embedding[y == int(i+1), :]
        new_x, new_y = embedding[y==i+1, 0], embedding[y==i+1, 1]
        plt.plot(new_x, new_y, 'o', color=colors, label=labels_prefix[i], markersize=3)
        plot_point_cov(pts, nstd=3, alpha=0.25, color=colors)
        xxx.append(new_x)
        yyy.append(new_y)
        
    # 添加坐标轴
    xxx_flat = np.concatenate(xxx)
    yyy_flat = np.concatenate(yyy)
    plt.xlim(np.min(xxx_flat)-x1, np.max(xxx_flat)+x2)
    plt.ylim(np.min(yyy_flat)-y1, np.max(yyy_flat)+y2)
    plt.xticks(size=5, color='black')
    plt.yticks(size=5, color='black')
    plt.xlabel('PC1 ({} %)'.format(round(model.explained_variance_ratio_[0] * 100, 2)), fontsize=6, color='black')
    plt.ylabel('PC2 ({} %)'.format(round(model.explained_variance_ratio_[1] * 100, 2)), fontsize=6, color='black')
    plt.legend(prop={"size": 5},  loc='upper right', frameon=False, edgecolor='none', facecolor='none')
    plt.savefig(f'{output}_plot1.svg', bbox_inches='tight', dpi=300)
    # plt.savefig(f'{output}_plot1.eps', bbox_inches='tight', dpi=300)
    # plt.show()
    plt.title(f'{output} Principal Component', size=8, color='black')
    # st.pyplot(plt.gcf())
    st.image(f'{output}_plot1.svg', use_column_width=True)

    # fig = px.line(selected_columns, title='Q-value vs Time [0-10 ns]')
    # fig.update_xaxes(title_text="Frames")
    # fig.update_yaxes(title_text="Q-value")
    # fig.update_layout(font=dict(family="Times New Roman", size=16, color='black'), legend_title="Groups")

# 2画方差值柱状图
def show_ratio(X, output):
    # st.subheader("Variance Ratio")
    model = PCA()
    embedding = model.fit_transform(X)

    cumulative_variance = []
    cumulative_sum = 0
    for ratio in list(model.explained_variance_ratio_):
        cumulative_sum += ratio*100
        cumulative_variance.append(cumulative_sum)

    PCA_var = pd.DataFrame(cumulative_variance)
    # PCA_var.to_csv(f"{output}_var.csv")
    
    plt.figure(dpi=300, figsize=(3.5, 3))
    plt.bar(range(len(cumulative_variance)), cumulative_variance, color = '#2d857a', width=0.35)
    for i in range(len(cumulative_variance)):
        plt.text(x=i, y=cumulative_variance[i]+0.5, s=str(round(cumulative_variance[i], 2)), ha='center', va='bottom', fontsize=5, color='black')
    plt.xticks(size=5, color='black')
    plt.yticks(size=5, color='black')
    plt.xlabel('Number of PCs', fontsize=6, color='black')
    plt.ylabel('Variance Ratio (%)', fontsize=6, color='black')
    plt.savefig(f'{output}_plot2.svg', bbox_inches='tight', dpi=300)
    # plt.savefig(f'{output}_plot2.eps', bbox_inches='tight', dpi=300)
    # plt.show()
    plt.title(f'{output} Variance Ratio', size=8, color='black')
    # st.pyplot(plt.gcf())
    st.image(f'{output}_plot2.svg', use_column_width=True)
    st.dataframe(PCA_var, height=300)


    # fig = px.line(selected_columns, title='Q-value vs Time [0-10 ns]')
    # fig.update_xaxes(title_text="Frames")
    # fig.update_yaxes(title_text="Q-value")
    # fig.update_layout(font=dict(family="Times New Roman", size=16, color='black'), legend_title="Groups")


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


# if __name__ == '__main__':
    # args = parameter()

    # data = pd.read_csv(args.f)
    # X = data.drop([args.n, args.g], axis=1)
    # labels, y = data[args.g], data[args.g]

    # show_ellipse(X, y, args.g, args.x1, args.x2, args.y1, args.y2, args.o)
    # show_ratio(X, args.o)

uploaded_file = st.sidebar.file_uploader("请上传csv数据文件 👇")

data = None
if uploaded_file is not None:
    data = load_data(uploaded_file)
elif st.sidebar.checkbox("或者加载案例数据", False):
    data = load_data('./data.csv')

if data is not None:
    aaa = st.columns(2)
    with aaa[0]:
        st.title(":green[**PCA**]")

        st.write(":red[*请注意，表格中的分组只能出现数字]")
        if st.checkbox("[可选] 显示加载数据集", False):
            st.dataframe(data, height=300, use_container_width=True)
        
        name = st.text_input("样本名称的列名name 👇", "name")
        group = st.text_input("样本分组的前缀legend 👇", "group")
        Title = st.text_input("输出图Title", "PCA")
        x1 = float(st.slider("x轴右-", 0, 100, 10))
        x2 = float(st.slider("x轴左+", 0, 100, 10))
        y1 = float(st.slider("y轴下+", 0, 100, 10))
        y2 = float(st.slider("y轴上-", 0, 100, 10))

    X = data.drop([name, group], axis=1)
    y = data[group]

    # if st.button('开始计算', type="primary", use_container_width=True):
    with aaa[1]:
        show_ellipse(X, y, group, x1, x2, y1, y2, Title)
        show_ratio(X, Title)

