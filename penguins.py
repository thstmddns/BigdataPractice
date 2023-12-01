import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')
X_train = pd.read_csv('./data/yemoonsaBigdata/datasets/Part2/penguin_X_train.csv')
y_train = pd.read_csv('./data/yemoonsaBigdata/datasets/Part2/penguin_y_train.csv')
X_test = pd.read_csv('./data/yemoonsaBigdata/datasets/Part2/penguin_X_test.csv')

# X_train.info()

# 데이터 전처리
train = pd.concat([X_train, y_train], axis=1)
# train.info()
# train.head()
train = train.dropna()
train.reset_index(inplace = True, drop = True)
X_train  = train[['species', 'island', 'sex', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']]
y_train  = train[['body_mass_g']]
# print(X_train.info())
# print(y_train.info())

# print(X_train.describe())

COL_NUM = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']
COL_CAT = ['species', 'island', 'sex']
COL_V = ['body_mass_g']

X = pd.concat([X_train, X_test])
ohe = OneHotEncoder()
ohe.fit(X[COL_CAT])

X_train_res = ohe.transform(X_train[COL_CAT])
X_test_res = ohe.transform(X_test[COL_CAT])

# print(X_train_res)

X_train_ohe = pd.DataFrame(X_train_res.todense(), columns = ohe.get_feature_names_out())
X_test_ohe = pd.DataFrame(X_test_res.todense(), columns = ohe.get_feature_names_out())

print(X_train_ohe)
X_train_fin = pd.concat([X_train[COL_NUM], X_train_ohe], axis = 1)
X_test_fin = pd.concat([X_test[COL_NUM], X_test_ohe], axis = 1)

X_tr, X_val, y_tr, y_val = train_test_split(X_train_fin, y_train, test_size=0.3)

scaler = MinMaxScaler()
scaler.fit(X_tr[COL_NUM])

X_tr[COL_NUM] = scaler.transform(X_tr[COL_NUM])
X_val[COL_NUM] = scaler.transform(X_val[COL_NUM])
X_test_fin[COL_NUM] = scaler.transform(X_test_fin[COL_NUM])

modelLR = LinearRegression()
modelLR.fit(X_tr, y_tr)

y_val_pred = modelLR.predict(X_val)
print(y_val_pred)