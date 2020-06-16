import pandas as pd
import numpy as np

# 删去缺省值超过一半的属性
def drop_missing_value(data):
    # 样本个数
    num_sample = data.shape[0]
    # 特征个数，除去最后一列标签
    # print(data.shape[1])
    num_attr = data.shape[1]-1
    data = data.values
    for i in range(num_attr):
        count_Nan = 0
        for j in range(num_sample):
            if np.isnan(data[j][i]):
                count_Nan = count_Nan+1
        Nan_rate = count_Nan/num_sample
        if Nan_rate >= 0.5:
            print("第"+str(i)+"列Nan占比大于0.5！")
            data = np.delete(data, i, axis=1)
        print("第"+str(i)+"列Nan占比", Nan_rate)

    data = pd.DataFrame(data, columns=None,index=None)
    data.to_csv("../arrhythmia_data/arrhythmia_drop_feature.csv",header=None,index=None)
    # print(len(data))
    # print(len(data[0]))
    return data

# 将剩下属性中缺失值小于一半的属性用众数、均值、中位数补齐
# 离散值用众数补齐，连续值用均值补齐
# 或者是sklearn.preprocessing.Imputer 用于对数据中的缺失值进行补全（线性差值）
def missing_value_complement(data):
    # 暂时先全部用众数补全
    # 存储每一列元素的众数
    data_mode = []
    for i in range(data.shape[1]-1):
        data_mode.append(list(data[i].mode()))

    # 使用numpy遍历
    data = data.values
    for i in range(len(data[0])):
        for j in range(len(data)):
            if np.isnan(data[j][i]):
                # 一列有多个众数时，用第一个元素即可
                data[j][i] = data_mode[i][0]

    # print(np.isnan(data[4][13]))
    # print(data_mode)
    # print(len(data_mode))
    # print(data)

    data_mode_complement = pd.DataFrame(data,columns=None,index=None)
    data_mode_complement.to_csv("../arrhythmia_data/arrhythmia_mode_complement.csv",header=None,index=None)
    return data_mode_complement

# 离散化数据
# 数据等频分组，参数data_list为一个排序好的列表，group_num为分组个数
def equal_frequency(data_list, group_num):
    # 每个组元素个数
    value_num = int(len(data_list)/group_num)
    # print(value_num)
    for i in range(0, len(data_list), value_num):
        for j in range(value_num):
            # 中位数作为分组区间所有元素的值
            data_list[i+j] = data_list[i+int(value_num/2)-1]
    return data_list

# 传入dataframe，将所有数据离散化
def discretization_data(data):
    global res
    row, column = data.shape
    # columns_name = data.columns
    for i in range(column):
        print("不同元素个数：", len(data[i].unique()))
        if len(data[i].unique()) > 1000:
            # 将数据集按指定列排序
            data = data.sort_values(by=i)
            data.index = range(len(data))
            a = np.array(data[[i]])
            a = a.reshape(len(a)).tolist()
            temp = equal_frequency(a, 100)
            temp = pd.DataFrame(temp)
            # 重新拼接数据集
            data = data.drop(i, axis=1)
            data = pd.concat([temp,data], axis=1)
            data = pd.DataFrame(data)
            # temp = np.array(temp).reshape(len(temp), 1)
            # data[columns_name[i]] = data[columns_name[i]].replace()
            # print(temp)
            # res.append(temp)
        else:
            print("离散属性！")
    return data

if __name__ == '__main__':
    # df = pd.read_csv("../arrhythmia_data/arrhythmia.csv", header=None)
    # df = drop_missing_value(df)
    # missing_value_complement(df)
    df = pd.read_csv("../data/experiment_data/all_data_with_label.csv", header=None)
    res1 = discretization_data(df)
    # print(res)
    res1.to_csv("../data/experiment_data/all_data_with_label_discretion.csv",index=None,header=None)
    ress = pd.read_csv("../data/experiment_data/all_data_with_label_discretion.csv",header=None)
    for i in range(ress.shape[1]):
        print(len(ress[i].unique()))
    print(ress.shape)
