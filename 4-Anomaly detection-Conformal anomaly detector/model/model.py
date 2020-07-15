import pandas as pd
from sklearn.metrics import mean_squared_error
import sys

sys.path.append('../alg/')
import conformal_anomaly_detector

# 超参设置
saved = True  # 是否保存文件
threshold = 0.2


# 预测某个文件的框架
def pred_csv(names, i_csv, paras, use_diff=False):
    data = pd.read_csv(f"../data/preprocessed/{i_csv + 1} preprocessed.csv",
                       dtype={'label': 'int', 'value_diff1': 'int', 'value': 'int'},
                       parse_dates=['timestamp'])
    x = data['value_diff1'].values if use_diff else data['value'].values

    # 预测
    print(paras[0], paras[1], use_diff)
    cad = conformal_anomaly_detector.Conformal(paras[0], paras[1])
    data['cad_score'] = 0  # 默认预测为不异常
    data['pre_label'] = 0  # 默认预测为不异常
    protect = []  # 保护期
    # is_protect = 0
    terrible_length = cad.L if cad.L > 50 else int(cad.L * 0.8)
    protect_length = cad.L * 4
    # 逐行预测每个score
    for t in range(cad.L + cad.M - 1, len(x)):
        # 不在保护期
        if t not in protect:
            # 计算保护期
            is_protect = 0
            for idx in range(terrible_length):
                if data.loc[t - 1 - idx, "pre_label"] == 0:
                    break
                else:
                    is_protect = is_protect + 1
            if is_protect == terrible_length:
                protect = [t + x for x in range(protect_length)]
                print(protect)

            p_t = cad.pred(t, x)
            data.loc[t, "cad_score"] = max(p_t - threshold, 0)  # 将预测出的概率减小threshold，但保证不小于0
            data.loc[t, "pre_label"] = 1 if p_t == 1 else 0
        # 在保护期内（不用计算新保护期了）
        else:
            p_t = 0.4 * (1 - ((t - min(protect)) / protect_length)) * (1 - threshold)  # 越远就可能性越小
            data.loc[t, "cad_score"] = max(p_t - threshold, 0)  # 将预测出的概率减小threshold，但保证不小于0
            data.loc[t, "pre_label"] = 0
    mse01 = mean_squared_error(data.label, data.cad_score)
    mse0 = mean_squared_error(data[data.label == 0].label, data[data.label == 0].cad_score)
    mse1 = mean_squared_error(data[data.label == 1].label, data[data.label == 1].cad_score)
    # 保存预测结果
    if saved:
        data.to_csv(f"../result/pred/{i_csv + 1}_pred_{use_diff}.csv", index=None, float_format='%.2f')
    return mse0, mse1, mse01


if __name__ == '__main__':
    names = ['htmjava_speed_7578_withlabel', 'htmjava_speed_t4013_withlabel', 'htmjava_TravelTime_387_withlabel',
             'htmjava_TravelTime_451_withlabel']
    paras_true = [[100, 60], [120, 70], [80, 50], [50, 30]]
    paras_false = [[40, 24], [40, 25], [80, 50], [50, 30]]
    for use_diff in [False, True]:
        result = pd.DataFrame(columns=('data', 'mse0', 'mse1', 'mse01'))
        result = result.astype({"mse0": float, "mse1": float, 'mse01': float})
        result['data'] = [1, 2, 3, 4]
        for i in range(len(names)):
            r_i = pred_csv(names, i, paras_true[i] if use_diff else paras_false[i], use_diff)
            print(names[i], r_i)
            result.loc[i, 'mse0'] = r_i[0]
            result.loc[i, 'mse1'] = r_i[1]
            result.loc[i, 'mse01'] = r_i[2]
        if saved:
            result.to_csv(f"../result/mse_{use_diff}.csv", index=None, float_format='%.3f')
