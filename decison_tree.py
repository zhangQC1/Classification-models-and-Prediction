#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
import graphviz
import pydotplus
from sklearn.svm import SVC
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier


def check_purity(data):
    '''Check if there is only one category'''
    '''检查当前数据是否只有一个类别'''
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False


def classify_data(data):
    '''classify the data'''
    '''做数据拆分'''
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]

    return classification


def get_potential_splits(data):
    '''Check the data, and find the set of all the ways splitting the data.'''
    '''遍历各个特征，获取可能拆分数据的方式集合'''
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):  # excluding the last column which is the label
        potential_splits[column_index] = []
        values = data[:, column_index]
        unique_values = np.unique(values)

        for index in range(len(unique_values)):
            if index != 0:
                current_value = unique_values[index]
                previous_value = unique_values[index - 1]
                potential_split = (current_value + previous_value) / 2

                potential_splits[column_index].append(potential_split)

    return potential_splits


def split_data(data, split_column, split_value):
    '''Split the data based on a feature.'''
    '''根据所选特征和所选值对数据进行拆分'''
    split_column_values = data[:, split_column]

    data_below = data[split_column_values <= split_value]
    data_above = data[split_column_values > split_value]

    return data_below, data_above


def calculate_entropy(data):
    '''calculate the entropy'''
    '''计算单个文件的熵'''
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy


def calculate_overall_entropy(data_below, data_above):
    '''calculate the overall entropy'''
    '''计算整体的熵'''
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy = (p_data_below * calculate_entropy(data_below)
                       + p_data_above * calculate_entropy(data_above))

    return overall_entropy


def determine_best_split(data, potential_splits):
    '''find the feature, and the best way to split the data'''
    '''寻找最佳的拆分特征及对应的拆分值'''
    overall_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value


def decision_tree_algorithm(df, counter=0, min_samples=20, max_depth=5):
    #Prepare the data
    # 数据准备
    if counter == 0:
        global COLUMN_HEADERS
        COLUMN_HEADERS = df.columns
        data = df.values

    else:
        data = df

    #The condition
    # 循环递归的条件
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)

        return classification


    # recursive part
    else:
        counter += 1

        #Get all the ways that can split the data
        # 获取所有可能的拆分
        potential_splits = get_potential_splits(data)
        #Get the best way
        # 寻找最有拆分方式
        split_column, split_value = determine_best_split(data, potential_splits)
        #split the data
        # 拆分数据
        data_below, data_above = split_data(data, split_column, split_value)

        #initial the tree
        # 初始化树
        feature_name = COLUMN_HEADERS[split_column]
        question = "{} <= {}".format(feature_name, split_value)
        sub_tree = {question: []}

        #Recursive to get the result
        # 递归寻找分类结果
        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth)

        #save the result
        # 保存结果
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree




# def deal_with_time(time_num):
#     result = datetime.datetime.fromtimestamp(time_num).strftime('%Y-%m-%d %H:%M:%S')
#     return result
#
#
# def get_features(tmp_df):
#     # get the depth and width
#     # 计算最大深度和最大广度
#     tmp_df['depth'] = 0
#     depth = 1
#     tmp_df.loc[0, 'depth'] = depth
#     id_list = tmp_df.loc[tmp_df['depth'] == depth, 'id'].values.tolist()
#     while len(id_list):
#         #         id_list = tmp_df.loc[tmp_df['depth']==depth,'id'].values.tolist()
#         cond = tmp_df['parent'].isin(id_list)
#         depth += 1
#         tmp_df.loc[cond, 'depth'] = depth
#         id_list = tmp_df.loc[tmp_df['depth'] == depth, 'id'].values.tolist()
#
#     max_depth = tmp_df['depth'].max()
#     max_range = tmp_df['depth'].value_counts().max()
#
#     #get the number of reposts on a day
#     # 计算一天内的转发量
#     tmp_df['t'] = pd.to_datetime(tmp_df['t'].apply(lambda x: deal_with_time(x)))
#     end_time = pd.to_datetime(tmp_df['t'].values[0]) + datetime.timedelta(1)
#     reposts_onaday = (tmp_df['t'] < end_time).sum()
#
#     feautures = list(tmp_df.loc[0, :].values)
#     feautures += [max_depth, max_range, reposts_onaday]
#     return feautures
### process the JSON(spend too long time, carefully)
### 处理json数据(运行时间很长，谨慎执行)
# folder_path = './weibo/Weibo/'
# print('文档的文件总数为:',len(os.listdir(folder_path)))
# #initial the aim
# #初始化目标dataframe
# # df = pd.DataFrame()
# features_list = []
# i = 0
# for file_name in os.listdir(folder_path):
#     i += 1
#     if i % 200 == 0:
#         print('已处理{}个文件'.format(i))
#     file_path = folder_path+file_name
#     try:
#         with open(file_path,encoding='utf-8') as f:
#             tmp_df = pd.DataFrame(json.load(f))
#             features = get_features(tmp_df)
#             features_list.append(features)
#     except:
#         print(file_name)
#         pass


if __name__ =='__main__':

    #get data
    #读取数据
    df = pd.read_excel('./data/processed_data.xlsx')

    #preprocess
    ### 数据处理
    #The length of the text
    # 新增text长度特征
    df['text_length'] = df['text'].apply(lambda x: len(str(x)))

    #whether has urls
    # 新增text中是否含有链接特征
    df['has_url'] = df['text'].apply(lambda x: 1 if 'http' in str(x) else 0)

    #whether has pictures
    # 处理是否有图片
    df['has_picture'] = df['picture'].apply(lambda x: 0 if str(x) == 'None' else 1)

    #Judge the verified
    # 处理verified字段
    df['verified'] = df['verified'].apply(lambda x: 1 if x else 0)

    #Judge the gender
    # 处理gender字段
    df['gender'] = df['gender'].apply(lambda x: 1 if x == 'f' else 0)

    #tackle the "rush_hour"
    df['t'] = df['t'].apply(lambda x: str(x))
    df['t'] = df['t'].apply(lambda x: 1 if (11 > int(x[11:13]) > 9 or 16 > int(x[11:13]) > 14 or 22 > int(x[11:13]) > 19) else 0)

    #tackle the data
    # 删除缺失值数据及转换数据类型
    df.dropna(
        subset=['bi_followers_count', 'attitudes_count', 'followers_count', 'comments_count', 'reposts_count'],
        inplace=True)
    for col in ['comments_count', 'bi_followers_count', 'attitudes_count']:
        df[col] = pd.to_numeric(df[col])
    #generate the aim
    # 创建目标字段
    df['label'] = ((df['reposts_count'] * 0.8 + df['comments_count']) >= 306.75) * 1

    #get the features
    # 选择建模特征
    features = ['bi_followers_count', 'verified', 'followers_count', 'friends_count', 'gender'
        , 'text_length', 'has_url', 'has_picture', 'max_depth', 'max_range', 'reposts_onaday', 't', 'attitudes_count']

    #build models
    # ## 数据建模
    # 定义输入输出
    X = df[features]
    y = df['label']
    #split the data set
    #拆分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data = pd.DataFrame(data=X_train,columns=features)
    train_data['label'] = y_train
    test_data = pd.DataFrame(data=X_test, columns=features)
    test_data['label'] = y_test

    #Decision tree
    ## 决策树建模
    #The code of the decision tree
    # 使用自己实现的算法生成决策树
    subtree = decision_tree_algorithm(train_data)

    clf = DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)
    #Print the result of the model
    #打印模型结果
    print('The accuracy of the decision tree (决策树模型预测的准确率):',accuracy_score(y_predict,y_test))
    print(classification_report(y_predict,y_test))
    dot_data = tree.export_graphviz(clf, out_file=None,
                            feature_names= features, # 特征名称
                            class_names=['No', 'Yes'], # 目标变量的类别名
                            filled=True, rounded=True,
                            special_characters=True) 
    graph = pydotplus.graph_from_dot_data(dot_data)
    # The view of the result
    #输出可视化结果
    graph.write_pdf('./decision_tree.pdf')


    #get the importances of features in the decision tree
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    num_features = len(importances)

    #The importance of the features
    # 将特征重要度以柱状图展示
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(num_features), importances[indices], color="g", align="center")
    plt.xticks(range(num_features), [features[i] for i in indices], rotation='45')
    plt.xlim([-1, num_features])
    plt.show()

    # Get the result of the importance
    # 输出各个特征的重要度
    for i in indices:
        print("{0} - {1:.3f}".format(features[i], importances[i]))


    # SVM model
    ## SVM建模
    svm = SVC()
    svm.fit(X_train,y_train)
    svm_predict = svm.predict(X_test)
    print('The prediction accuracy of the SVM:', accuracy_score(svm_predict, y_test))
    print(classification_report(svm_predict, y_test))

    #random forests
    forest = RandomForestClassifier()
    forest.fit(X_train,y_train)
    forest_predict = forest.predict(X_test)
    print('The prediction accuracy of the random forests:', accuracy_score(forest_predict, y_test))
    print(classification_report(forest_predict, y_test))

