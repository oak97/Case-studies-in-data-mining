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


# 用alphas计算w
def cal_ws(alphas, X, y):
    data_mat = mat(X)
    label_mat = mat(y).transpose()
    m, n = data_mat.shape
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * label_mat[i], data_mat[i, :].T)
    return w


# 6-6
def kernel_trans(X, A, kTup):
    m = X.shape[0]
    K = mat(zeros((m, 1)))
    if kTup[0] == 'linear':
        K = X * A.T  # linear kernel
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))  # divide in NumPy is element-wise not matrix like Matlab
    else:
        raise NameError('We Have a Problem -- That Kernel is not recognized')
    return K


class optStruct:
    def __init__(self, data_mat, class_labels, C, toler, kTup):
        self.X = data_mat
        self.label_mat = class_labels
        self.C = C
        self.tol = toler
        self.m = data_mat.shape[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.e_cache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernel_trans(self.X, self.X[i, :], kTup)


# 6-7
def innerL(i, optS):
    E_i = cal_Ek(optS, i)
    if ((optS.label_mat[i] * E_i < -optS.tol) and (optS.alphas[i] < optS.C)) \
            or ((optS.label_mat[i] * E_i > optS.tol) and (optS.alphas[i] > 0)):
        j, E_j = select_j(i, optS, E_i)  # changed
        alpha_i_old = optS.alphas[i].copy()
        alpha_j_old = optS.alphas[j].copy()
        if (optS.label_mat[i] != optS.label_mat[j]):
            L = max(0, optS.alphas[j] - optS.alphas[i])
            H = min(optS.C, optS.C + optS.alphas[j] - optS.alphas[i])
        else:
            L = max(0, optS.alphas[j] + optS.alphas[i] - optS.C)
            H = min(optS.C, optS.alphas[j] + optS.alphas[i])
        if L == H:
            # print ("L==H")
            return 0
        eta = 2.0 * optS.K[i, j] - optS.K[i, i] - optS.K[j, j]  # changed for kernel
        if eta >= 0:
            # print ("eta>=0")
            return 0
        optS.alphas[j] -= optS.label_mat[j] * (E_i - E_j) / eta
        optS.alphas[j] = clip_alpha(optS.alphas[j], L, H)
        update_E_k(optS, j)
        if (abs(optS.alphas[j] - alpha_j_old) < 0.00001):
            # print ("j not moving enough")
            return 0
        optS.alphas[i] += optS.label_mat[j] * optS.label_mat[i] * (alpha_j_old - optS.alphas[j])
        update_E_k(optS, i)
        b1 = optS.b - E_i - optS.label_mat[i] * (optS.alphas[i] - alpha_i_old) * optS.K[i, i] \
             - optS.label_mat[j] * (optS.alphas[j] - alpha_j_old) * optS.K[i, j]  # changed for kernel
        b2 = optS.b - E_j - optS.label_mat[i] * (optS.alphas[i] - alpha_i_old) * optS.K[i, j] \
             - optS.label_mat[j] * (optS.alphas[j] - alpha_j_old) * optS.K[j, j]  # changed for kernel
        if (0 < optS.alphas[i]) and (optS.C > optS.alphas[i]):
            optS.b = b1
        elif (0 < optS.alphas[j]) and (optS.C > optS.alphas[j]):
            optS.b = b2
        else:
            optS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def cal_Ek(optS, k):
    f_xk = float(multiply(optS.alphas, optS.label_mat).T * optS.K[:, k] + optS.b)
    E_k = f_xk - float(optS.label_mat[k])
    return E_k


def update_E_k(optS, k):
    E_k = cal_Ek(optS, k)
    optS.e_cache[k] = [1, E_k]


def select_j(i, optS, E_i):
    max_k = -1
    max_delta_E = 0
    E_j = 0
    optS.e_cache[i] = [1, E_i]
    valid_e_cache_list = nonzero(optS.e_cache[:, 0].A)[0]
    if (len(valid_e_cache_list) > 1):
        for k in valid_e_cache_list:
            if k == i: continue
            E_k = cal_Ek(optS, k)
            delta_E = abs(E_i - E_k)
            if (delta_E > max_delta_E):
                max_k = k
                max_delta_E = delta_E
                E_j = E_k
        return max_k, E_j
    else:
        j = select_j_rdm(i, optS.m)
        E_j = cal_Ek(optS, j)
    return j, E_j


def smoP(data_mat, class_labels, C, toler, max_iter, kTup=('linear', 0)):  # 默认linear
    optS = optStruct(mat(data_mat), mat(class_labels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alpha_pairs_changed = 0
    while (iter < max_iter) and ((alpha_pairs_changed > 0) or (entireSet)):
        alpha_pairs_changed = 0
        if entireSet:  # go over all
            for i in range(optS.m):
                alpha_pairs_changed += innerL(i, optS)
                # print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alpha_pairs_changed))
            iter += 1
        else:  # go over non-bound (railed) alphas
            nonBoundIs = nonzero((optS.alphas.A > 0) * (optS.alphas.A < C))[0]
            for i in nonBoundIs:
                alpha_pairs_changed += innerL(i, optS)
                # print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alpha_pairs_changed))
            iter += 1
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif (alpha_pairs_changed == 0):
            entireSet = True
            # print ("iteration number: %d" % iter)
    return optS.b, optS.alphas


# "rbf",1.3,200,0.0001,1000
def predict(X, y, ktype='linear', k1=0, C=0.6, toler=0.0001, max_iter=40):
    b, alphas = smoP(X, y, C, toler, max_iter, (ktype, k1))
    data_mat = mat(X)
    label_mat = mat(y).transpose()
    sv_idx = nonzero(alphas.A > 0)[0]
    sv_data = data_mat[sv_idx]
    sv_label = label_mat[sv_idx]
    print("there are %d Support Vectors" % sv_data.shape[0])
    m, n = data_mat.shape
    y_pred = mat(zeros((m, 1)))
    prob_pre = mat(zeros((m, 1)))
    w = zeros((n, 1))
    for i in range(m):
        kernel_eval = kernel_trans(sv_data, data_mat[i, :], (ktype, k1))
        prob_pre[i] = kernel_eval.T * multiply(sv_label, alphas[sv_idx]) + b
        y_pred[i] = sign(prob_pre[i])
        w += multiply(alphas[i] * label_mat[i], data_mat[i, :].T)
    return y_pred, prob_pre, w, b, alphas, sv_idx, sv_data, sv_label
