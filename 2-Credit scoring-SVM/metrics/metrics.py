import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family']=['Songti SC']
plt.rcParams['axes.unicode_minus'] = False

# 下面是对模型进行评估，自己实现计算和画图
# 我们已有X、y、y_predict，显然可以计算混淆矩阵，本次作业中不划分数据集
# - 查全率、召回率：recall，宁愿错杀，不可漏掉
# - 查准率、精确率：precision，宁愿漏掉，不可错杀
# - KS系数
# TPR真正例率 TP/(TP+FN)
# FPR假正例率 FP/(FP+TN)
# 每个都是自己拟出一个y_predict，然后一样，计算出结果

class metricStruct:
    def __init__(self,l0,l1):
        self.l0 = l0
        self.l1 = l1

def cal_TFPN(df,mStruct):
    pd.DataFrame.very_deep_copy = very_deep_copy
    dff = df.very_deep_copy()
    dff["confusion"] = cvt(dff["y"],dff["y_predict"])
    TP_cvt = cvt(mStruct.l1,mStruct.l1)
    FN_cvt = cvt(mStruct.l1,mStruct.l0)
    FP_cvt = cvt(mStruct.l0,mStruct.l1)
    TN_cvt = cvt(mStruct.l0,mStruct.l0)
    dff.insert(3, "TP", (dff["confusion"]==TP_cvt).astype("int"))
    dff.insert(3, "FN", (dff["confusion"]==FN_cvt).astype("int"))
    dff.insert(3, "FP", (dff["confusion"]==FP_cvt).astype("int"))
    dff.insert(3, "TN", (dff["confusion"]==TN_cvt).astype("int"))
    TP,FN,FP,TN = dff[["TP","FN","FP","TN"]].sum()
    return TP,FN,FP,TN

def recall_and_precision(df,mStruct):
    pd.DataFrame.very_deep_copy = very_deep_copy
    dff = df.very_deep_copy()
    TP,FN,FP,TN = cal_TFPN(dff,mStruct)
    # print(TP,FN,FP,TN)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    return recall, precision

def cal_ks(df,cnt,mStruct):
    pd.DataFrame.very_deep_copy = very_deep_copy
    dff = df.very_deep_copy()
    dfff = dff.sort_values("prob_predict", ascending=False)
    #刷新y_predict
    dfff["y_predict"] = np.concatenate(( np.ones(cnt),
                                         np.zeros(dfff.shape[0]-cnt) if mStruct.l0==0 else -np.ones(dfff.shape[0]-cnt) ),axis=0)
    #刷新TN FP FN TP
    TP,FN,FP,TN = cal_TFPN(dfff,mStruct)
    # print(f"TP,FN,FP,TN = {TP,FN,FP,TN}")
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    return TPR,FPR,TPR-FPR

def cal_max_ks(df,mStruct):
    pd.DataFrame.very_deep_copy = very_deep_copy
    dff = df.very_deep_copy()
    CNT = dff.shape[0]
    TPR_list = [0] * CNT
    FPR_list = [0] * CNT
    KS_list = [0] * CNT
    for cnt in range(CNT):
        TPR_list[cnt],FPR_list[cnt],KS_list[cnt] = cal_ks(dff,cnt+1,mStruct)
    KS_loc = np.argmax(KS_list)
    KS_val = max(KS_list)
    return KS_val,KS_loc,TPR_list,FPR_list,KS_list

def plot_ks_curve(df,mStruct):
    pd.DataFrame.very_deep_copy = very_deep_copy
    dff = df.very_deep_copy()
    CNT = dff.shape[0]
    KS_val,KS_loc,TPR,FPR,KS = cal_max_ks(dff,mStruct)
    fig, ax = plt.subplots()
    ax.plot(np.array(range(1,CNT+1))/CNT,TPR, 'go',label='真正例率', linewidth=1)
    ax.plot(np.array(range(1,CNT+1))/CNT,FPR, 'bo',label='假正例率', linewidth=1)
    ax.set(xlabel='判定比例', ylabel='真正例率 或 假正例率', title='KS曲线')
    ax.grid()
    plt.vlines(KS_loc/CNT, FPR[KS_loc], TPR[KS_loc], colors = "r", linestyles = "dotted",linewidth=4,label ='最大距离')
    ax.legend()
    plt.show()
    return KS_val
    

#构造映射函数，加速运算
def cvt(y,yp): 
    return (y+10)*(yp+2)

def very_deep_copy(self):
    return pd.DataFrame(self.values.copy(), self.index.copy(), self.columns.copy())