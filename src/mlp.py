import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

train_data.dropna(how='all', inplace=True)

train_feature = train_data.drop(
    ["personal id", "loan default", "year of birth", "credit score"], axis=1).to_numpy()
train_label = train_data["loan default"].to_numpy()

test_feature = test_data.drop(["personal id", "year of birth", "credit score"],
                              axis=1).to_numpy()

m = 0
n = 0

drop_line = []
for i in train_feature:
    n = 0
    for j in i:
        if j == float('inf'):
            # print("m=", m, "n=", n)
            list.append(drop_line, m)
        elif j is None:
            list.append(drop_line, m)
        n = n + 1
    m = m + 1

train_feature = np.delete(train_feature, drop_line, axis=0)
train_label = np.delete(train_label, drop_line, axis=0)

m = 0
n = 0
for i in test_feature:
    n = 0
    for j in i:
        if j == float('inf'):
            test_feature[m][n] = np.finfo(np.float32).max
            # test_feature[m][n] = 0
        elif j is None:
            test_feature[m][n] = 0
        n = n + 1
    m = m + 1

feature, predict, feature_res, predict_res = train_test_split(train_feature, train_label,
                                                              test_size=0.2, random_state=0)
mlp = MLPClassifier(hidden_layer_sizes=[150, 150, 150], activation='logistic', alpha=0.00001)
T1 = time.time()
mlp.fit(feature, feature_res)
res = mlp.predict(predict)
T2 = time.time()
print('MLP')
print('f1-score: ' + str(f1_score(res, predict_res, average='macro')))
print('process time: ' + str(T2 - T1))
