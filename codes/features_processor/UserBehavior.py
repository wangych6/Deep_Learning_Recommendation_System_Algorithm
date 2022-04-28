#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 19:46
# @Email   : chestorwang@tencent.com
# @File    : UserBehavior.py
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

FILE_ROOT = '../datasets/'  # 数据集的路径
MAX_LEN = 50  # DIN使用的历史行为序列长度


def user_behavior():
    data_raw = pd.read_csv(FILE_ROOT + 'UserBehavior.csv', names=["user_id", "item_id", "cate_id", "behavior", "time"])
    # 对行为特征进行转换
    le = LabelEncoder()
    data_raw["behavior"] = le.fit_transform(data_raw["behavior"])
    # 以CTR为例
    data_ctr = data_raw[data_raw["behavior"] == 0]

    data = data_ctr
    # 对sparse特征进行labelencode，方便后续的emebdding
    sparse_features = ["user_id", "item_id", "cate_id"]
    for fea in sparse_features:
        le = LabelEncoder()
        data[fea] = le.fit_transform(data[fea])

    # 0一般作为序列特征的padding标记，所以对序列特征 id+1
    data["item_id"] = data["item_id"] + 1
    data["cate_id"] = data["cate_id"] + 1

    # 获得统计信息
    n_users, n_items, n_cates = data["user_id"].nunique(), data["item_id"].nunique(), data["cate_id"].nunique()

    # 生成历史序列的list
    data = data.sort_values(['user_id', 'time']).groupby('user_id').agg(
        click_hist_list=('item_id', list),
        cate_hist_hist=('cate_id', list)
    ).reset_index()

    # 划窗生成正负样本
    train_data, val_data, test_data = [], [], []
    for item in data.itertuples():
        if len(item[2]) < 10:
            continue
        click_hist_list = item[2][:MAX_LEN]
        cate_hist_list = item[3][:MAX_LEN]
        hist_list = []

        def neg_sample():
            neg = click_hist_list[0]
            while neg in click_hist_list:
                neg = random.randint(1, n_items) # 全局随机生成一条负样本
            return neg

        neg_list = [neg_sample() for _ in range(len(click_hist_list))]

        for i in range(1, len(click_hist_list)):
            hist_list.append([click_hist_list[i - 1], cate_hist_list[i - 1]])
            if i == len(click_hist_list) - 1:
                test_data.append([hist_list.copy(), [click_hist_list[i], cate_hist_list[i]], 1])
                test_data.append([hist_list.copy(), [neg_list[i], cate_hist_list[i]], 0])
            if i == len(click_hist_list) - 2:
                val_data.append([hist_list.copy(), [click_hist_list[i], cate_hist_list[i]], 1])
                val_data.append([hist_list.copy(), [neg_list[i], cate_hist_list[i]], 0])
            else:
                train_data.append([hist_list.copy(), [click_hist_list[i], cate_hist_list[i]], 1])
                train_data.append([hist_list.copy(), [neg_list[i], cate_hist_list[i]], 0])

    # shuffle
    # random.shuffle(train_data)
    # random.shuffle(val_data)
    # random.shuffle(test_data)
    # print('shuffle True')

    train = pd.DataFrame(train_data, columns=['click_hist', 'target_item', 'label'])
    val = pd.DataFrame(val_data, columns=['click_hist', 'target_item', 'label'])
    test = pd.DataFrame(test_data, columns=['click_hist', 'target_item', 'label'])

    # padding到maxlen
    train_X = [np.array([0.] * len(train)), np.array([0] * len(train)),
               pad_sequences(train['click_hist'], maxlen=MAX_LEN),
               np.array(train['target_item'].tolist())]
    train_y = train['label'].values
    val_X = [np.array([0] * len(val)), np.array([0] * len(val)),
             pad_sequences(val['click_hist'], maxlen=MAX_LEN),
             np.array(val['target_item'].tolist())]
    val_y = val['label'].values
    test_X = [np.array([0] * len(test)), np.array([0] * len(test)),
              pad_sequences(test['click_hist'], maxlen=MAX_LEN),
              np.array(test['target_item'].tolist())]
    test_y = test['label'].values

    # feature columns
    feature_columns = [[],[{'feat': 'item_id', 'feat_num': n_items+1, 'embed_dim': 8}]]  # sparseFeature('cate_id', cate_count, embed_dim)

    # behavior
    behavior_list = ['item_id']  # , 'cate_id'
    return feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (test_X, test_y)

if __name__ == '__main__':
    feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (test_X, test_y) =  user_behavior()
    print(train_X, train_y)
