import pandas as pd
import os
import numpy as np

# 传感器物理属性应该单独列出进行分类
# 几个不同的粒度：煤矿的物理属性 > 传感器的物理属性 > 传感器的实时数据

# 传感器物理属性提取
def chuanganqi_physics_feature_generate():
    df = pd.read_csv("../data/partition/YiKuang_partition/YiKuang.csv",encoding="gbk")
    df = df[["pointId","typeName","range","upperCut","upperAlarm"]]
    df = df.drop_duplicates()
    df.to_csv("../data/chuanganqi_information/yikuang_physics_feature.csv",index=None,encoding="gbk")

# 煤矿物理属性特征提取，暂时未加入
def mine_feature_extraction():
    df = pd.read_csv()


# 实时浓度数据特征提取
def nongdu_feature_extraction(id):
    # 加入传感器的物理属性
    physics_feature = pd.read_csv("../data/partition/YiKuang_partition/yikuang_chuanganqi_E.csv",encoding="gbk")
    physics_feature.columns = ["pointId", "upperAlarm", "partition"]
    # 找到与函数传入参数id相对应的upperAlarm值
    upperAlarm_value = physics_feature.loc[physics_feature["pointId"] == id.astype(int), ["upperAlarm"]]
    upperAlarm_value = upperAlarm_value.iloc[0, 0]

    filePath = str("../data/YiKuang/partition_E/")+str(id)+str("/2018.xlsx")
    if os.path.exists(filePath):
        df = pd.read_excel(filePath, header=0)

        # date属性中截取日期到小时，可以自定义时间窗口
        # 没有做时间飘移
        df["TimeScale"] = df["TimeScale"].map(lambda x: str(x)[0:13]+str(":00:00"))
        # 列表存放提取的特征
        list_feature = []

        # 按天分割时间
        divide_by_day = df["TimeScale"].unique()
        # 只提取有十五天以上记录的传感器特征
        if len(divide_by_day) >= 15:
            for day in divide_by_day:
                # 提取原始数据中规定时间窗口内所有数据到temp_data
                temp_data = df[df["TimeScale"].isin([day])]
                # 提取最大值（maxValue）属性的最大值、最小值、均值、方差
                maxValue_max = temp_data["maxValue"].max()
                maxValue_min = temp_data["maxValue"].min()
                maxValue_avg = temp_data["maxValue"].mean()
                maxValue_var = temp_data["maxValue"].var()

                # 提取最小值（minValue）属性的最大值、最小值、均值、方差
                minValue_max = temp_data["minValue"].max()
                minValue_min = temp_data["minValue"].min()
                minValue_avg = temp_data["minValue"].mean()
                minValue_var = temp_data["minValue"].var()

                # 提取均值（avgValue）属性的最大值、最小值、均值、方差
                avgValue_max = temp_data["avgValue"].max()
                avgValue_min = temp_data["avgValue"].min()
                avgValue_avg = temp_data["avgValue"].mean()
                avgValue_var = temp_data["avgValue"].var()

                # 均值之和，最大值的均值与均值的平均值之差（maxValue_avg - avgValue_avg）
                # avg的平均值与最小值的均值之差（avgValue_avg - minValue_avg）
                avgValue_sum = temp_data["avgValue"].sum()
                max_avg_sub = maxValue_avg - avgValue_avg
                avg_min_sub = avgValue_avg - minValue_avg

                # 实际最大浓度减去上限报警浓度、瓦斯浓度最大值与最小值的差、瓦斯浓度最大值与最小值的和
                # 最大浓度减去上限报警浓度
                over_upperAlarm_value = maxValue_max - upperAlarm_value

                # 上限报警浓度减去最小浓度
                under_upperAlarm_minvalue = upperAlarm_value - minValue_min

                # 规定时间窗口内瓦斯浓度maxValue属性中最大值与最小值的和与差
                max_nongdu_subtract = maxValue_max - maxValue_min
                max_nongdu_add = maxValue_max + maxValue_min

                # 规定时间窗口内瓦斯浓度minValue属性中最大值与最小值的和与差
                min_nongdu_subtract = minValue_max - minValue_min
                min_nongdu_add = minValue_max + minValue_min

                # 规定时间窗口内瓦斯浓度maxValue属性和minValue属性中各自最大值的和与差
                max_maxmin_nongdu_subtract = maxValue_max - minValue_max
                max_maxmin_nongdu_add = maxValue_max + minValue_max

                # 规定时间窗口内瓦斯浓度maxValue属性和minValue属性中各自最小值的和与差
                min_maxmin_nongdu_subtract = maxValue_min - minValue_min
                min_maxmin_nongdu_add = maxValue_min + minValue_min

                # 规定时间窗口内浓度超限次数
                upperAlarm_count = (temp_data["maxValue"] >= upperAlarm_value).astype(int).sum()

                # 规定时间窗口内浓度超限浓度之和
                upperAlarm_nongdu_sum = temp_data[["maxValue"]][temp_data["maxValue"] >= upperAlarm_value]["maxValue"].sum()

                # 创建分类时的标签，发生浓度超限为1，没有为零。作为Label放在最后一列
                if upperAlarm_count > 0:
                     label = 1
                else:
                    label = 0

                # print(str(day), "maxValue_max:",maxValue_max)
                # print(str(day), "maxValue_min:",maxValue_min)
                # print(str(day), "maxValue_avg:",maxValue_avg)
                # print(str(day), "maxValue_var:",maxValue_var)
                #
                # print(str(day), "minValue_max:",minValue_max)
                # print(str(day), "minValue_min:",minValue_min)
                # print(str(day), "minValue_avg:", minValue_avg)
                # print(str(day), "minValue_var:", minValue_var)
                # print("---------------------------")

                list_feature.append(list(
                    [day,maxValue_max,maxValue_min,maxValue_avg,maxValue_var,minValue_max,minValue_min,minValue_avg,minValue_var,
                     over_upperAlarm_value,max_nongdu_subtract,max_nongdu_add,min_nongdu_subtract,min_nongdu_add,
                     max_maxmin_nongdu_subtract,max_maxmin_nongdu_add,min_maxmin_nongdu_subtract,min_maxmin_nongdu_add,
                     avgValue_max,avgValue_min,avgValue_avg,avgValue_var,avgValue_sum,max_avg_sub,avg_min_sub,under_upperAlarm_minvalue,
                     upperAlarm_nongdu_sum,upperAlarm_count,label]))

            # 转化为dataframe
            list_feature = pd.DataFrame(list_feature,columns=["date","maxValue_max","maxValue_min","maxValue_avg","maxValue_var",
                        "minValue_max","minValue_min","minValue_avg","minValue_var","over_upperAlarm_value","max_nongdu_subtract",
                        "max_nongdu_add","min_nongdu_subtract","min_nongdu_add","max_maxmin_nongdu_subtract","max_maxmin_nongdu_add",
                        "min_maxmin_nongdu_subtract","min_maxmin_nongdu_add","avgValue_max","avgValue_min","avgValue_avg",
                        "avgValue_var","avgValue_sum","max_avg_sub","avg_min_sub","under_upperAlarm_minvalue",
                        "upperAlarm_nongdu_sum", "upperAlarm_count", "Label"])

            # 按日期排序
            list_feature = list_feature.sort_values(by="date")

            # # 排序特征：某一天瓦斯浓度最大值、最小值、均值、方差的排序特征
            # # 瓦斯浓度最大值浓度排序
            # df1 = list_feature.sort_values(by="maxValue_max", ascending=False)
            # # 排序后要重置索引
            # df1.index = range(len(df1))
            # list_feature = pd.concat([df1, pd.DataFrame(list(range(1, list_feature.shape[0] + 1)), columns=["maxValue_max_rank"])],
            #                          axis=1).sort_values(by="date")
            #
            # # 瓦斯浓度最小值浓度排序
            # df1 = list_feature.sort_values(by="maxValue_min", ascending=False)
            # # 排序后要重置索引
            # df1.index = range(len(df1))
            # list_feature = pd.concat([df1, pd.DataFrame(list(range(1, list_feature.shape[0] + 1)), columns=["maxValue_min_rank"])],
            #                          axis=1).sort_values(by="date")
            #
            # # 瓦斯浓度最大值均值排序
            # df1 = list_feature.sort_values(by="maxValue_avg", ascending=False)
            # # 排序后要重置索引
            # df1.index = range(len(df1))
            # list_feature = pd.concat([df1, pd.DataFrame(list(range(1, list_feature.shape[0] + 1)), columns=["maxValue_avg_rank"])],
            #                          axis=1).sort_values(by="date")
            #
            # # 瓦斯浓度最大值方差排序
            # df1 = list_feature.sort_values(by="maxValue_var", ascending=False)
            # # 排序后要重置索引
            # df1.index = range(len(df1))
            # list_feature = pd.concat([df1, pd.DataFrame(list(range(1, list_feature.shape[0] + 1)), columns=["maxValue_var_rank"])],
            #                          axis=1).sort_values(by="date")
            #
            # # 瓦斯浓度最小值浓度排序
            # df1 = list_feature.sort_values(by="minValue_max", ascending=False)
            # # 排序后要重置索引
            # df1.index = range(len(df1))
            # list_feature = pd.concat([df1, pd.DataFrame(list(range(1, list_feature.shape[0] + 1)), columns=["minValue_max_rank"])],
            #                          axis=1).sort_values(by="date")
            #
            # # 瓦斯浓度最小值浓度排序
            # df1 = list_feature.sort_values(by="minValue_min", ascending=False)
            # # 排序后要重置索引
            # df1.index = range(len(df1))
            # list_feature = pd.concat([df1, pd.DataFrame(list(range(1, list_feature.shape[0] + 1)), columns=["minValue_min_rank"])],
            #                          axis=1).sort_values(by="date")
            #
            # # 瓦斯浓度最小值均值排序
            # df1 = list_feature.sort_values(by="minValue_avg", ascending=False)
            # # 排序后要重置索引
            # df1.index = range(len(df1))
            # list_feature = pd.concat([df1, pd.DataFrame(list(range(1, list_feature.shape[0] + 1)), columns=["minValue_avg_rank"])],
            #                          axis=1).sort_values(by="date")
            #
            # # 瓦斯浓度最小值方差排序
            # df1 = list_feature.sort_values(by="minValue_var", ascending=False)
            # # 排序后要重置索引
            # df1.index = range(len(df1))
            # list_feature = pd.concat([df1, pd.DataFrame(list(range(1, list_feature.shape[0] + 1)), columns=["minValue_var_rank"])],
            #                          axis=1).sort_values(by="date")
            #
            # # 每天瓦斯超限次数在全年中的排名
            # df1 = list_feature.sort_values(by="upperAlarm_count", ascending=False)
            # # 排序后要重置索引
            # df1.index = range(len(df1))
            # list_feature = pd.concat([df1, pd.DataFrame(list(range(1, list_feature.shape[0] + 1)), columns=["upperAlarm_count_rank"])],
            #                          axis=1).sort_values(by="date")

            # # 调整位置，将upperAlarm_count作为标签移到最后一列,回归标签列
            # upperAlarm_count = list_feature.pop("upperAlarm_count")
            # # upperAlarm_count = upperAlarm_count.astype(int)
            # list_feature.insert(list_feature.shape[1],"upperAlarm_count",upperAlarm_count)
            #
            # # 调整位置，将Label作为标签移到最后一列，分类标签列
            # label = list_feature.pop("Label")
            # list_feature.insert(list_feature.shape[1],"Label",label)

            # 创建保存数据的文件夹,第一次运行时需要创建文件夹，之后需要注释掉
            # dirPath = str("../data/featureExtraction/YiKuang/partition_E/")+str(id)
            # os.mkdir(dirPath)

            # 保存数据
            filePath_save = str("../data/featureExtraction/YiKuang/partition_E/")+str(id)+ str("/") + str(id) + str(
                        "_featureExtraction.csv")
            list_feature.to_csv(filePath_save, index=None)
        else:
            print("数据过少")
            return
    else:
        print("传感器 "+str(id)+" 没有2018年记录")
        return

# 提取规定时间窗口内瓦斯超限次数，timeWindows_num参数控制时间窗口个数
def upperAlarm_count_before_day(df,timeWindows_num):
    df1 = df["upperAlarm_count"]
    list_before = []
    list_upperAlarm_count = list(df1)
    for i in range(len(list_upperAlarm_count)):
        if i < timeWindows_num:
            list_before.append(0)
        else:
            list_before.append(sum(list_upperAlarm_count[i - timeWindows_num:i]))
    return list_before

def concat_upperAlarm_count_data_before_day(id):
    filePath_read = str("../data/featureExtraction/YiKuang/partition_E/") + str(id) + str("/") + str(id) + str(
        "_featureExtraction.csv")
    if os.path.exists(filePath_read):
        list_feature = pd.read_csv(filePath_read, header=0)
        # print(df)

        # 提取提取当日前几天天的瓦斯超限次数，gap_day参数控制时间间隔
        # 提取当日前一天瓦斯浓度超限次数
        list_before = upperAlarm_count_before_day(list_feature,1)
        # print(list_before)
        list_feature = pd.concat([list_feature,pd.DataFrame(list_before,columns=["upperAlarm_count_before_1_day"])],axis=1)

        # 提取当日前三天瓦斯浓度超限次数
        list_before = upperAlarm_count_before_day(list_feature,3)
        # print(list_before)
        list_feature = pd.concat([list_feature,pd.DataFrame(list_before,columns=["upperAlarm_count_before_3_day"]).reset_index()],axis=1)

        # 提取当日前五天瓦斯浓度超限次数
        list_before = upperAlarm_count_before_day(list_feature,5)
        # print(list_before)
        list_feature = pd.concat([list_feature,pd.DataFrame(list_before,columns=["upperAlarm_count_before_5_day"])],axis=1)

        # 提取当日前十天天瓦斯浓度超限次数
        list_before = upperAlarm_count_before_day(list_feature,10)
        # print(list_before)
        list_feature = pd.concat([list_feature,pd.DataFrame(list_before,columns=["upperAlarm_count_before_10_day"])],axis=1)

        # 提取当日前十五天天瓦斯浓度超限次数
        list_before = upperAlarm_count_before_day(list_feature,15)
        # print(list_before)
        list_feature = pd.concat([list_feature,pd.DataFrame(list_before,columns=["upperAlarm_count_before_15_day"])],axis=1)

        # 保存拼接后的文件
        list_feature = pd.DataFrame(list_feature)
        filePath_save = str("../data/featureExtraction/YiKuang/partition_E/") + str(id) + str("/") + str(id) + str(
            "_featureExtraction.csv")
        list_feature = list_feature.drop("index",axis=1)

        # # 调整位置，将upperAlarm_count作为标签移到最后一列,回归标签列
        # upperAlarm_count = list_feature.pop("upperAlarm_count")
        # # upperAlarm_count = upperAlarm_count.astype(int)
        # list_feature.insert(list_feature.shape[1], "upperAlarm_count", upperAlarm_count)
        #
        # # 调整位置，将Label作为标签移到最后一列，分类标签列
        # label = list_feature.pop("Label")
        # list_feature.insert(list_feature.shape[1], "Label", label)

        # 保存到CSV
        list_feature.to_csv(filePath_save,index=None)
    else:
        return

# 提取规定时间窗口的前几个时间窗口的瓦斯超限的最大值、最小值、均值、方差，timeWindows_num参数控制时间窗口个数
def tongji_feature_before_day(df,timeWindows_num):
    df1 = df["maxValue_max"]
    list_before_max = []
    list_before_min = []
    list_before_avg = []
    list_before_var = []
    # list_upperAlarm_count = list(df1)
    for i in range(len(df1)):
        if i < timeWindows_num:
            list_before_max.append(0)
            list_before_min.append(0)
            list_before_avg.append(0)
            list_before_var.append(0)
        else:
            list_before_max.append(df1[i - timeWindows_num:i].max())
            list_before_min.append(df1[i - timeWindows_num:i].min())
            list_before_avg.append(df1[i - timeWindows_num:i].mean())
            list_before_var.append(df1[i - timeWindows_num:i].var())
    return list_before_max, list_before_min, list_before_avg, list_before_var

def concat_tongji_data_before_day(id):
    filePath_read = str("../data/featureExtraction/YiKuang/partition_E/") + str(id) + str("/") + str(id) + str(
        "_featureExtraction.csv")
    if os.path.exists(filePath_read):
        list_feature = pd.read_csv(filePath_read, header=0)

        # 提取maxValue属性前三个规定时间窗口的最大值、最小值、均值、方差
        list_before_max,list_before_min,list_before_avg,list_before_var = tongji_feature_before_day(list_feature, 3)
        list_feature = pd.concat([list_feature, pd.DataFrame(list_before_max, columns=["maxValue_max_before_3_day"]),
                   pd.DataFrame(list_before_min, columns=["maxValue_min_before_3_day"]),
                   pd.DataFrame(list_before_avg, columns=["maxValue_avg_before_3_day"]),
                   pd.DataFrame(list_before_var, columns=["maxValue_var_before_3_day"])], axis=1)

        # 提取maxValue属性前五个规定时间窗口的最大值、最小值、均值、方差
        list_before_max,list_before_min,list_before_avg,list_before_var = tongji_feature_before_day(list_feature,5)
        list_feature = pd.concat([list_feature, pd.DataFrame(list_before_max, columns=["maxValue_max_before_5_day"]),
                   pd.DataFrame(list_before_min, columns=["maxValue_min_before_5_day"]),
                   pd.DataFrame(list_before_avg, columns=["maxValue_avg_before_5_day"]),
                   pd.DataFrame(list_before_var, columns=["maxValue_var_before_5_day"])], axis=1)

        # 提取maxValue属性前十个规定时间窗口的最大值、最小值、均值、方差
        list_before_max,list_before_min,list_before_avg,list_before_var = tongji_feature_before_day(list_feature,10)
        list_feature = pd.concat([list_feature, pd.DataFrame(list_before_max, columns=["maxValue_max_before_10_day"]),
                   pd.DataFrame(list_before_min, columns=["maxValue_min_before_10_day"]),
                   pd.DataFrame(list_before_avg, columns=["maxValue_avg_before_10_day"]),
                   pd.DataFrame(list_before_var, columns=["maxValue_var_before_10_day"])], axis=1)

        # 提取maxValue属性前十五个规定时间窗口的最大值、最小值、均值、方差
        list_before_max,list_before_min,list_before_avg, list_before_var = tongji_feature_before_day(list_feature,15)
        list_feature = pd.concat([list_feature, pd.DataFrame(list_before_max, columns=["maxValue_max_before_15_day"]),
                   pd.DataFrame(list_before_min, columns=["maxValue_min_before_15_day"]),
                   pd.DataFrame(list_before_avg, columns=["maxValue_avg_before_15_day"]),
                   pd.DataFrame(list_before_var, columns=["maxValue_var_before_15_day"])], axis=1)

        # 保存拼接后的文件
        list_feature = pd.DataFrame(list_feature)
        filePath_save = str("../data/featureExtraction/YiKuang/partition_E/") + str(id) + str("/") + str(id) + str(
            "_featureExtraction.csv")
        # list_feature = list_feature.drop("index", axis=1)

        # 调整位置，将upperAlarm_count作为标签移到最后一列,回归标签列
        upperAlarm_count = list_feature.pop("upperAlarm_count")
        # upperAlarm_count = upperAlarm_count.astype(int)
        list_feature.insert(list_feature.shape[1], "upperAlarm_count", upperAlarm_count)

        # 调整位置，将Label作为标签移到最后一列，分类标签列
        label = list_feature.pop("Label")
        list_feature.insert(list_feature.shape[1], "Label", label)

        # 保存到CSV
        list_feature.to_csv(filePath_save, index=None)
    else:
        return

# 提取传感器之间关联特征
# 同一区域内前几天所有传感器超限次数
def interPoints_feature_before_day(gap_day,id_current):
    partition = pd.read_csv("../data/partition/YiKuang_partition/YiKuang_chuanganqi_E.csv")
    partition = list(partition.iloc[:,0])
    partition.remove(id_current)

    filePath_read = str("../data/featureExtraction/YiKuang/partition_E/") + str(id_current) + str("/") + str(id_current) + str(
            "_featureExtraction.csv")
    if os.path.exists(filePath_read):
        df_current = pd.read_csv(filePath_read, header=0)
        list_partition_upperAlarm_count_before_day = np.zeros(len(df_current))
    else:
        return

    for id in partition:
        filePath_read = str("../data/featureExtraction/YiKuang/partition_E/") + str(id) + str("/") + str(id) + str(
            "_featureExtraction.csv")
        if os.path.exists(filePath_read):
            df = pd.read_csv(filePath_read,header=0)
            column_name = str("upperAlarm_count_before_")+str(gap_day)+str("_day")
            df1 = df[column_name]

            # 当前传感器（id_current）记录天数比循环里传感器（id）记录天数多，循环里传感器（id）记录后面补零
            if len(df_current)>len(df):
                append_zero = np.zeros(len(df_current)-len(df1))
                list_partition_upperAlarm_count_before_day = np.array(list_partition_upperAlarm_count_before_day)+\
                                                         np.concatenate((np.array(df1),append_zero))

            # 当前传感器（id_current）记录天数比循环里传感器（id）记录天数少，循环里传感器（id）记录后面剪切
            else:
                list_partition_upperAlarm_count_before_day = np.array(list_partition_upperAlarm_count_before_day)+\
                                                             np.array(df1)[0:len(df_current)]
    return list_partition_upperAlarm_count_before_day

# 合并数据
def concat_interpPoints_feature_before_day(id):
    filePath_read = str("../data/featureExtraction/YiKuang/partition_E/") + str(id) + str("/") + str(id) + str(
        "_featureExtraction.csv")
    if os.path.exists(filePath_read):
        list_feature = pd.read_csv(filePath_read, header=0)

        # 提取同一区域内前一天传感器超限次数之和
        list_partition_upperAlarm_count_before_1_day = interPoints_feature_before_day(1, id)

        # 提取同一区域内前一天传感器超限次数之和
        list_partition_upperAlarm_count_before_3_day = interPoints_feature_before_day(3, id)

        # 提取同一区域内前一天传感器超限次数之和
        list_partition_upperAlarm_count_before_5_day = interPoints_feature_before_day(5, id)

        # 提取同一区域内前一天传感器超限次数之和
        list_partition_upperAlarm_count_before_10_day = interPoints_feature_before_day(10, id)

        # 提取同一区域内前一天传感器超限次数之和
        list_partition_upperAlarm_count_before_15_day = interPoints_feature_before_day(15, id)

        # 将提取的特征拼接到所有特征集里
        list_feature = pd.concat([list_feature,pd.DataFrame(list_partition_upperAlarm_count_before_1_day,columns=["interPoints_upperAlarm_count_before_1_day"]),
                                  pd.DataFrame(list_partition_upperAlarm_count_before_3_day,columns=["interPoints_upperAlarm_count_before_3_day"]),
                                  pd.DataFrame(list_partition_upperAlarm_count_before_5_day,columns=["interPoints_upperAlarm_count_before_5_day"]),
                                  pd.DataFrame(list_partition_upperAlarm_count_before_10_day,columns=["interPoints_upperAlarm_count_before_10_day"]),
                                  pd.DataFrame(list_partition_upperAlarm_count_before_15_day,columns=["interPoints_upperAlarm_count_before_15_day"])],axis=1)

        # 保存拼接后的文件
        list_feature = pd.DataFrame(list_feature)
        filePath_save = str("../data/featureExtraction/YiKuang/partition_E/") + str(id) + str("/") + str(id) + str(
            "_featureExtraction.csv")
        # list_feature = list_feature.drop("index", axis=1)

        # 调整位置，将upperAlarm_count作为标签移到最后一列,回归标签列
        upperAlarm_count = list_feature.pop("upperAlarm_count")
        # upperAlarm_count = upperAlarm_count.astype(int)
        list_feature.insert(list_feature.shape[1], "upperAlarm_count", upperAlarm_count)

        # 调整位置，将Label作为标签移到最后一列，分类标签列
        label = list_feature.pop("Label")
        list_feature.insert(list_feature.shape[1], "Label", label)

        # 保存到CSV
        list_feature.to_csv(filePath_save, index=None)
    else:
        return

# 提取同一区域内前几天浓度最大值（maxValue属性）的统计特征
def interPoint_tongji_feature_before_day(gap_day,id):
    partition = pd.read_csv("../data/partition/YiKuang_partition/YiKuang_chuanganqi_E.csv")
    partition = list(partition.iloc[:,0])
    partition.remove(id)

    filePath_read = str("../data/featureExtraction/YiKuang/partition_E/") + str(id) + str("/") + str(id) + str(
        "_featureExtraction.csv")
    if os.path.exists(filePath_read):
        df = pd.read_csv(filePath_read, header=0)
        list_partition_upperAlarm_count_before_day = []
    else:
        return

    for id in partition:
        filePath_read = str("../data/featureExtraction/YiKuang/partition_E/") + str(id) + str("/") + str(id) + str(
            "_featureExtraction.csv")
        if os.path.exists(filePath_read):
            df = pd.read_csv(filePath_read,header=0)
            df1 = df["maxValue_max"]

if __name__ == '__main__':
    partiton = pd.read_csv("../data/partition/YiKuang_partition/YiKuang_chuanganqi_E.csv")
    partiton = list(partiton.iloc[:,0])
    for id in partiton:
        print("---正在提取传感器 " + str(id) + " 特征---")
        nongdu_feature_extraction(id)
        concat_upperAlarm_count_data_before_day(id)
        concat_tongji_data_before_day(id)
        # 有问题，没有按日期对齐
        # concat_interpPoints_feature_before_day(id)
        print("---提取完成---")
