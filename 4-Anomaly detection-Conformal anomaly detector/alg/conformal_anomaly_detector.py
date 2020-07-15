import pandas as pd
import numpy as np
from sklearn.neighbors._ball_tree import BallTree
from sklearn.metrics import mean_squared_error
import logging

import sys

sys.path.append('../util/')
import log


class Conformal(object):
    def __init__(self, L=40, M=20):
        self.L = L  # 因为L大于20，所以knn用的球树
        assert self.L >= 20, "L要大于20，knn才用的球树"
        self.M = M
        self.T = int(self.M * 0.6)
        self.C = self.M - self.T
        self.k = int(self.T * 0.5)

    def pred(self, t, x):
        # t = 80
        x_t = x[t]
        # print(x_t)
        Z = np.zeros((self.L, self.M), dtype=np.int16)

        for l in range(self.M):
            Z[:, l] = x[t - self.L - self.M + 1 + l:t - self.M + 1 + l]

        Z_T = Z[:, :self.T]
        Z_C = Z[:, self.T:]
        a = np.zeros(self.C)

        def NCM(o, Z_T):
            Y = np.c_[o, Z_T]
            # print(Y.T[:1])
            tree = BallTree(Y.T, leaf_size=3)
            dist, ind = tree.query(Y.T[:1], k=self.k + 1)
            # print(ind)  # indices of k closest neighbors
            # print(dist)  # distances to k closest neighbors
            # print(dist.sum())
            return dist.sum()

        for l in range(self.C):
            o = Z_C[:, l]
            a[l] = NCM(o, Z_T)

        z = x[t - self.L + 1:t + 1]
        assert z[-1] == x_t, "新观测点错误"
        a_t = NCM(z, Z_T)
        # print(a_t)
        p_t = (a - a_t < 0).astype(int).sum() / self.C
        return p_t


# 测试一个文件
if __name__ == '__main__':
    names = ['htmjava_speed_7578_withlabel', 'htmjava_speed_t4013_withlabel', 'htmjava_TravelTime_387_withlabel',
             'htmjava_TravelTime_451_withlabel']
    data = pd.read_csv(f"../data/{names[3]}.csv", usecols=[1, 2, 3], dtype={'label': 'int', 'value': 'int'},
                       parse_dates=['timestamp'])
    x = data['value'].values

    # 预测
    cad = Conformal()
    data['cad_score'] = 0  # 前面的都预测为不异常
    for t in range(cad.L + cad.M - 1, len(x)):
        p_t = cad.pred(t, x)
        # print(t, x[t], p_t)
        # print("---------")
        data.loc[t, "cad_score"] = max(p_t - 0.2, 0)

    mse01 = mean_squared_error(data.label, data.cad_score)
    mse0 = mean_squared_error(data[data.label == 0].label, data[data.label == 0].cad_score)
    mse1 = mean_squared_error(data[data.label == 1].label, data[data.label == 1].cad_score)
