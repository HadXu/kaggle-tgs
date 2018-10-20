import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img

train = pd.read_csv('../input/train.csv', index_col='id', usecols=[0])
depths = pd.read_csv('../input/depths.csv', index_col='id')
train = train.join(depths)

ids_test = depths[~depths.index.isin(train.index)].index.values
pd.DataFrame({'id': ids_test}).to_csv('../input/test_ids.csv', index=False)

train['images'] = [np.array(load_img('../input/train/images/{}.png'.format(idx), grayscale=True)) / 255 for idx in
                   train.index]
train['masks'] = [np.array(load_img('../input/train/masks/{}.png'.format(idx), grayscale=True)) / 255 for idx in
                  train.index]

train['image_sum'] = train['images'].apply(np.sum)
print(train[train['image_sum'] == 0].shape[0])
train = train.drop(train[train['image_sum'] == 0].index)

train['coverage'] = train.masks.map(np.sum) / pow(101, 2)


def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i


train['coverage_class'] = train.coverage.map(cov_to_class)


def cov_to_class_v2(val):
    for i in range(0, 11):
        if val / 100 <= i:
            return i


train['z_class'] = train.z.map(cov_to_class_v2)

train['class'] = train['z_class'] * 100 + train['coverage_class']

less_num = train['class'].value_counts()[(train['class'].value_counts() < 10)].index.values
less_num_dict = dict(zip(less_num, (less_num / 100).astype(int) * 10000))
train.loc[train['class'].isin(less_num), 'class'] = train.loc[train['class'].isin(less_num), 'class'].map(less_num_dict)

ids_train, ids_valid = train_test_split(
    train.index.values,
    test_size=0.2, stratify=train['class'], random_state=1926)

train.reset_index(drop=False)[['id', 'class']].to_csv('../input/data_ids_with_class.csv', index=False)
