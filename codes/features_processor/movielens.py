# __author__: wangych
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import pandas as pd

FILE_ROOT = './datasets/'

def MovieLens():
    # 读取数据
    fileroot = conf.movie_lens
    movies = pd.read_csv(fileroot+'movies.csv')
    rating = pd.read_csv(fileroot+'ratings.csv')
    data = pd.merge(rating, movies, on='movieId')

    # 构造用户平均分和电影平均分作为新特征
    data['userMean'] = data.groupby(by='userId')['rating'].transform('mean')
    data['movieMean'] = data.groupby(by='movieId')['rating'].transform('mean')

    # 转换为二分类问题
    data.loc[(data.rating>=data.userMean), 'label'] = 1
    data.loc[(data.rating<data.userMean), 'label'] = 0

    print(data)

    # 特征工程
    features = ['userId', 'movieId', 'timestamp', 'genres', 'userMean', 'movieMean']
    cate_feature = ['genres']
    for feat in cate_feature:
        enc = LabelEncoder()
        data[feat] = enc.fit_transform(data[feat])
    # 切分数据集
    data = sklearn.utils.shuffle(data)
    train, test = train_test_split(data, test_size=0.2)

    X_train = train[features].values.astype('int32')
    y_train = train['label'].values.astype('int32')
    X_test = test[features].values.astype('int32')
    y_test = test['label'].values.astype('int32')



def main():
    create_ml_datasets()

if __name__ == '__main__':
    main()
