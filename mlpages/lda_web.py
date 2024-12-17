import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import cm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os, zipfile, warnings, argparse
import streamlit as st
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Arial'] + plt.rcParams['font.serif']

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

def show_ellipse(X, y, prefix, x1, x2, y1, y2, output):
    model = LinearDiscriminantAnalysis(n_components=2)
    embedding = model.fit_transform(X, y)

    unique_drugs = set(y)
    regions = sorted(list(unique_drugs))

    labels_prefix = [f"{prefix}{region}" for region in regions]

    color_map = plt.get_cmap('rainbow')
    color_indices = np.linspace(0, 1, len(regions))

    # 定义分辨率
    plt.figure(dpi=300, figsize=(3.5, 3))

    LDA1, LDA2 = [], []
    for i in range(0, len(regions), 1):
        colors = color_map(color_indices[i])
        pts = embedding[y == int(i+1), :]
        new_x, new_y = embedding[y==i+1, 0], embedding[y==i+1, 1]
        plt.plot(new_x, new_y, 'o', color=colors, label=labels_prefix[i], markersize=3)
        plot_point_cov(pts, nstd=3, alpha=0.25, color=colors)
        LDA1.append(new_x)
        LDA2.append(new_y)
        
    # 添加坐标轴
    xxx_flat = np.concatenate(LDA1)
    yyy_flat = np.concatenate(LDA2)
    plt.xlim(np.min(xxx_flat)-x1, np.max(xxx_flat)+x2)
    plt.ylim(np.min(yyy_flat)-y1, np.max(yyy_flat)+y2)
    plt.xticks(size=5, color='black')
    plt.yticks(size=5, color='black')
    plt.xlabel('LDA1 ({} %)'.format(round(model.explained_variance_ratio_[0] * 100, 2)), fontsize=6, color='black')
    plt.ylabel('LDA2 ({} %)'.format(round(model.explained_variance_ratio_[1] * 100, 2)), fontsize=6, color='black')
    plt.legend(prop={"size": 5},  loc='upper right', frameon=False, edgecolor='none', facecolor='none')
    plt.savefig(f'{output}_plot1.svg', bbox_inches='tight', dpi=300)
    plt.title(f'{output}', size=7, color='black')
    st.image(f'{output}_plot1.svg', use_container_width=True)

    return LDA1, LDA2


def show_ratio(X, output):
    model = LinearDiscriminantAnalysis()
    embedding = model.fit_transform(X, y)

    cumulative_variance = []
    cumulative_sum = 0
    for ratio in list(model.explained_variance_ratio_):
        cumulative_sum += ratio*100
        cumulative_variance.append(cumulative_sum)

    LDA_var = pd.DataFrame(cumulative_variance, columns=['LDA Cumulative Variance'])

    plt.figure(dpi=300, figsize=(3.5, 3))
    plt.bar(range(len(cumulative_variance)), cumulative_variance, color = '#2d857a', width=0.35)
    for i in range(len(cumulative_variance)):
        plt.text(x=i, y=cumulative_variance[i]+0.5, s=str(round(cumulative_variance[i], 2)), ha='center', va='bottom', fontsize=5, color='black')
    plt.xticks(size=5, color='black')
    plt.yticks(size=5, color='black')
    plt.xlabel('Number of LDAs', fontsize=6, color='black')
    plt.ylabel('Variance Ratio (%)', fontsize=6, color='black')
    plt.savefig(f'{output}_plot2.svg', bbox_inches='tight', dpi=300)
    plt.title(f'{output}', size=7, color='black')
    
    st.image(f'{output}_plot2.svg', use_container_width=True)
    st.dataframe(LDA_var, height=300, use_container_width=True)

def add_stat_annotation(ax, **kws):
    r2 = kws.pop('r2', None)
    pval = kws.pop('pval', None)
    if r2 is not None:
        ax.text(0.05, 0.95, f'$R^2$={r2:.2f}', transform=ax.transAxes, fontsize=15, verticalalignment='top')
    if pval is not None:
        ax.text(0.05, 0.90, f'p-value={pval:.2e}', transform=ax.transAxes, fontsize=15, verticalalignment='top')

def plot_concentration(data, output):
    slope, intercept, r_value, p_value, std_err = stats.linregress(data['LDA1'], data['Concentration'])
    g = sns.lmplot(x='LDA1', y='Concentration', data=data, palette=['#bb2649'])
    add_stat_annotation(g.ax, r2=r_value**2, pval=p_value)
    plt.savefig(f'{output}_plot3.svg', bbox_inches='tight', dpi=300)
    st.image(f'{output}_plot3.svg', use_container_width=True)

@st.cache_resource
def load_data(Inputfile):
    return pd.read_csv(Inputfile)

uploaded_file = st.sidebar.file_uploader("请上传csv数据文件 👇")

data = None
if uploaded_file is not None:
    data = load_data(uploaded_file)
elif st.sidebar.checkbox("或者加载案例数据", False):
    data = load_data('./data.csv')

if data is not None:
    aaa = st.columns(2)
    with aaa[0]:
        st.title(":green[**LDA**]")

        st.write(":red[*请注意，表格中的分组只能出现数字]")
        if st.checkbox("[可选] 显示上传数据", True):
            st.dataframe(data, height=300, use_container_width=True)

        name = st.text_input("样本名称的列名name 👇", "name")
        group = st.text_input("样本分组的前缀legend 👇", "group")
        Title = st.text_input("输出图Title", "LDA")
        Conc = st.text_input("如果需要绘制LDA1与浓度的曲线，填写浓度所在列名", "Concentration")

        x1 = float(st.number_input("x轴下限- [可以填入负数]"))
        x2 = float(st.number_input("x轴上限+"))
        y1 = float(st.number_input("y轴下限-"))
        y2 = float(st.number_input("y轴上限+"))

    X = data.drop([name, group], axis=1)
    y = data[group]

    with aaa[1]:
        LDA1, LDA2 = show_ellipse(X, y, group, x1, x2, y1, y2, Title)
        show_ratio(X, Title)

        if Conc in data.columns:
            data2 = pd.DataFrame({'LDA1': np.concatenate(LDA1), 'Concentration': data[Conc]})
            plot_concentration(data2, Title)
            st.dataframe(data2, use_container_width=True)
