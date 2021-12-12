import time
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np

clfs = [XGBClassifier(colsample_bytree=0.75, n_estimators=1000),
        # RandomForestClassifier(max_depth=150, max_features=35),
        DecisionTreeClassifier(criterion='gini', max_depth=500, splitter="random", class_weight='balanced',
                               min_impurity_decrease=0.000002)]

# clfs = [
#         RandomForestClassifier(max_depth=150, max_features=35),
#         DecisionTreeClassifier(criterion='gini', max_depth=500, splitter="random", class_weight='balanced',
#                                min_impurity_decrease=0.000002)]

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


m = 0
n = 0
for i in train_feature:
    n = 0
    for j in i:
        if j == float('inf'):
            train_feature[m][n] = np.finfo(np.float32).max
            # test_feature[m][n] = 0
        elif j is None:
            train_feature[m][n] = 0
        n = n + 1
    m = m + 1

# X, X_predict, y, y_predict = train_test_split(train_feature, train_label, test_size=0.33, random_state=2017)
X = train_feature
y = train_label
X_predict = test_feature

dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_predict.shape[0], len(clfs)))

n_folds = 5
skf = StratifiedKFold(n_folds)
for j, clf in enumerate(clfs):
    '''依次训练各个单模型'''
    o = 0
    for (train, test) in skf.split(X, y):
        o += 1
    dataset_blend_test_j = np.zeros((X_predict.shape[0], o+1))
    i = -1
    for (train, test) in skf.split(X, y):
        i += 1
        X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
        clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:, 1]
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict_proba(X_predict)[:, 1]
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
    # print("val auc Score: %f" % f1_score(y_predict, dataset_blend_test[:, j]))
clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
clf.fit(dataset_blend_train, y)
y_submission = clf.predict_proba(dataset_blend_test)[:, 1]

y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

for i in y_submission:
    if i >= 0.5:
        print(1)
    else:
        print(0)
# print("blend result")
# print("val auc Score: %f" % (f1_score(y_predict, y_submission)))

# res = clf.predict(test_feature)
# print(res)
# res = (res - res.min()) / (res.max()-res.min())
# for i in res:
#     print(i)
