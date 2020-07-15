import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analysis(data, need_prepro, csv_i):
    v = 'value_diff1' if need_prepro else 'value'
    timeseries = data[v]
    plt.figure(figsize=(12, 8))
    plt.plot(timeseries, color='k', label='Original', alpha=0.2, linewidth=2)
    plt.scatter(data[(data['label'] == '1') & (data['pre_label'] == '0')].index,
                data[(data['label'] == '1') & (data['pre_label'] == '0')][v], c='r', marker='o',
                alpha=0.4, label='Only Original 1')
    plt.scatter(data[(data['pre_label'] == '1') & (data['label'] == '0')].index,
                data[(data['pre_label'] == '1') & (data['label'] == '0')][v], c='g', marker='o',
                alpha=0.4, label='Only Predict 1')
    plt.scatter(data[(data['pre_label'] == '1') & (data['label'] == '1')].index,
                data[(data['pre_label'] == '1') & (data['label'] == '1')][v],
                c='blue', marker='s',
                alpha=0.4, label='Both Original and Predict 1')
    plt.legend(loc='best')
    strr = ' after stationarity' if need_prepro else ""
    plt.title(f"Prediction result of Data {csv_i + 1}" + strr)

    # 保存图片
    plt.savefig(f'../result/pics/Predict/{csv_i + 1}.png', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    names = ['htmjava_speed_7578_withlabel', 'htmjava_speed_t4013_withlabel', 'htmjava_TravelTime_387_withlabel',
             'htmjava_TravelTime_451_withlabel']
    need_prepro = [True, False, False, False]

    for csv_i in range(len(names)):
        data = pd.read_csv(f"../result/pred/{csv_i + 1}_pred_{need_prepro[csv_i]}.csv",
                           dtype={'label': 'category', 'pre_label': 'category', 'value': 'int', 'value_diff1': "int"},
                           parse_dates=['timestamp'])
        analysis(data, need_prepro[csv_i], csv_i)
