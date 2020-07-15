import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

plt.rcParams['font.family'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.options.display.max_rows = 200

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=100, suppress=True)
from statsmodels.tsa.seasonal import seasonal_decompose


def plot_line(data1, name, type='label'):
    plt.figure(figsize=(20, 5))
    plt.plot(data1['timestamp'], data1['value'], 'k', alpha=0.2, linewidth=2)

    plt.scatter(data1[data1[type] == '0']['timestamp'], data1[data1[type] == '0']['value'], c='g', marker='.',
                alpha=0.4)
    plt.scatter(data1[data1[type] == '1']['timestamp'], data1[data1[type] == '1']['value'], c='r', marker='.',
                alpha=0.4)

    plt.ylabel('value')
    plt.xlabel('timestamp')
    plt.title(name)

    plt.grid(True)
    plt.savefig(f'../result/pics/{name}.png', dpi=200, bbox_inches='tight')
    plt.show()


# 时间序列稳定化测试
def test_stationarity(data, window_num, name, labeled=False):
    timeseries = data['value_diff1']
    # Determing rolling statistics
    rolmean = timeseries.rolling(window=window_num).mean()
    rolstd = timeseries.rolling(window=window_num).std()

    # Plot rolling statistics:
    fig = plt.figure(figsize=(16, 8))
    orig = plt.plot(timeseries, color='blue', label='Original', alpha=0.2, linewidth=2)
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='green', label='Rolling Std')
    plt.legend(loc='best')
    plt.title(f'First Difference of {name}')

    # 在图上标注平稳性信息
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    dfoutput['Window Length'] = window_num
    plt.text(1.08 * len(timeseries), 0, f'Results of Dickey-Fuller Test:\n'
                                        f'{dfoutput}'
                                        f'\n\nConclusion:'
                                        f'\np-value < 0.05, Test Statistic < Critical Value (1%)\nTherefore, the time series is stable after first-order difference.')
    # 保存图片
    plt.savefig(f'../result/pics/Diffed/{csv_i + 1} First Difference.png', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    names = ['htmjava_speed_7578_withlabel', 'htmjava_speed_t4013_withlabel', 'htmjava_TravelTime_387_withlabel',
             'htmjava_TravelTime_451_withlabel']
    # r_names = list(x + "_pred" for x in names)

    # # 查看原始图
    # for name in names:
    #     data = pd.read_csv(f"../data/{name}.csv", usecols=[1, 2, 3], dtype={'label': 'category', 'value': 'int'},
    #                        parse_dates=['timestamp'])
    #     plot_line(data, name, 'label')
    #
    # # 查看预测图
    # for name in r_names:
    #     data = pd.read_csv(f"../result/{name}.csv", dtype={'pre_label': 'category', 'value': 'int'},
    #                        parse_dates=['timestamp'])
    #     plot_line(data, name, 'pre_label')

    # 窗口参数
    # 时间窗长度
    freqs = {0: 110, 1: 140, 2: 15, 3: 50}
    for csv_i in range(len(names)):
        data = pd.read_csv(f"../data/{names[csv_i]}.csv",
                           dtype={'label': 'category', 'value': 'int'},
                           parse_dates=['timestamp'])
        # 粗略查看默认分解情况
        decomposition = seasonal_decompose(data['value'], freq=freqs[csv_i])
        decomposition.plot()
        plt.savefig(f'../result/pics/Decomposition/{csv_i + 1} Decomposition.png', dpi=200, bbox_inches='tight')
        # 平稳处理测试
        data['value_diff1'] = data['value'].diff(1).fillna(method='bfill').astype('int')
        test_stationarity(data, freqs[csv_i], names[csv_i])
        # 保存预处理后的结果
        data.to_csv(f"../data/preprocessed/{csv_i + 1} preprocessed.csv", index=None,
                    float_format='%.2f')
