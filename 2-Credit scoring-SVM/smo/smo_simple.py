import random
from numpy import *


# 6-1
def select_j_rdm(i, m):
    j = i
    while (j == i):
        j = random.randint(0, m - 1)
    return j


def clip_alpha(alpha, L, H):
    if alpha > H:
        alpha = H
    elif alpha < L:
        alpha = L
    return alpha


# 6-2
def smoSimple(X, y, C, toler, MAX_ITER):
    data_mat = mat(X)  # 1412*13
    label_mat = mat(y).transpose()  # 1412*1
    m = data_mat.shape[0]
    alphas = mat(zeros((m, 1)))  # 1412*1，这就是SMO求出的\alpha向量，向量中非零点对应的样本就是支持向量
    b = 0
    iter = 0
    while (iter < MAX_ITER):
        alpha_pairs_changed = 0
        for i in range(m):
            f_xi = float(multiply(alphas, label_mat).T *
                         (data_mat * data_mat[i, :].T)) + b
            E_i = f_xi - float(label_mat[i])
            if ((label_mat[i] * E_i < -toler) and (alphas[i] < C)) or \
                    ((label_mat[i] * E_i > toler) and (alphas[i] > 0)):
                j = select_j_rdm(i, m)
                f_xj = float(multiply(alphas, label_mat).T *
                             (data_mat * data_mat[j, :].T)) + b
                E_j = f_xj - float(label_mat[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                if (label_mat[i] != label_mat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if (L == H):
                    # print("L==H")
                    continue
                eta = 2.0 * data_mat[i, :] * data_mat[j, :].T \
                      - data_mat[i, :] * data_mat[i, :].T - \
                      data_mat[j, :] * data_mat[j, :].T
                if eta >= 0:
                    # print("eta>=0")
                    continue
                alphas[j] -= label_mat[j] * (E_i - E_j) / eta
                alphas[j] = clip_alpha(alphas[j], L, H)
                if (abs(alphas[j] - alpha_j_old) < 0.000001):
                    # print("j not moving enough")
                    continue
                alphas[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])
                b1 = b - E_i - \
                     label_mat[i] * (alphas[i] - alpha_i_old) * data_mat[i, :] * data_mat[i, :].T - \
                     label_mat[j] * (alphas[j] - alpha_j_old) * \
                     data_mat[i, :] * data_mat[j, :].T
                b2 = b - E_j - \
                     label_mat[i] * (alphas[i] - alpha_i_old) * data_mat[i, :] * data_mat[j, :].T - \
                     label_mat[j] * (alphas[j] - alpha_j_old) * \
                     data_mat[j, :] * data_mat[j, :].T
                if (0 < alphas[i] and (C > alphas[i])):
                    b = b1
                elif (0 < alphas[j] and (C > alphas[j])):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
                # print(f"iter={iter}, i={i}, pairs changed {alpha_pairs_changed} times.")
        if (alpha_pairs_changed == 0):
            iter += 1
        else:
            iter = 0
        # print(f"iteration number = {iter}")
    return b, alphas


def cal_ws(alphas, X, y):
    data_mat = mat(X)
    label_mat = mat(y).transpose()
    m, n = data_mat.shape
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * label_mat[i], data_mat[i, :].T)
    return w


def predict(X, y, max_iter=40, C=0.6, toler=0.0001):
    b, alphas = smoSimple(X, y, C, toler, max_iter)
    data_mat = mat(X)
    label_mat = mat(y).transpose()
    sv_idx = nonzero(alphas.A > 0)[0]
    sv_data = data_mat[sv_idx]
    sv_label = label_mat[sv_idx]
    print("there are %d Support Vectors" % sv_data.shape[0])
    ws = cal_ws(alphas, X, y)
    m, n = data_mat.shape
    y_pred = mat(zeros((m, 1)))
    prob_pre = mat(zeros((m, 1)))
    for i in range(m):
        prob_pre[i] = data_mat[i] * mat(ws) + b
        y_pred[i] = sign(prob_pre[i])
    return y_pred, prob_pre, ws, b, alphas, sv_idx, sv_data, sv_label
