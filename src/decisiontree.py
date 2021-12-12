import time

from predict import *
import matplotlib.pyplot as plt
import seaborn as sns

solution = Predict()
solution.PreProcess()

res = {}
mmax = 0
max_d = 0
max_t = 0
# data = np.empty((9, 7))
onelist = range(1, 100)
# onelist = [0.000001, 0.000002, 0.000003, 0.000004, 0.000005, 0.000006, 0.000007]
# onelist = [0.000001, 0.000002]
# for g in onelist:
#     res[g] = solution.Refine('gini', 170, 'random', 'balanced', g)

m = 0
n = 0
# T1 = time.time()
# res = solution.Refine('gini', 150, 'random', 'balanced', 0.000002)
# T2 = time.time()
# print('decision tree')
# print('f1-score: ' + str(res))
# print('process time: ' + str(T2-T1))
#
data = {}
for d in range(20, 200, 20):
    for t in onelist:
        tmp = solution.Refine('gini', d, 'random', 'balanced', t/1000000)
        if not str(d) in data.keys():
            data[str(d)] = {}
            data[str(d)][str(t/1000000)] = tmp
        else:
            data[str(d)][str(t/1000000)] = tmp
        # data[m][n] = tmp
        n += 1
    m += 1
#
# ax = sns.heatmap(pd.DataFrame(data), annot=True)
ax = sns.heatmap(pd.DataFrame(data))
plt.show()
# print(max_d)
#
# print(max_t)
# print(mmax)
# while t >= min_purity_decrease_right:
#     res[t] = solution.Refine('entropy', 100, 'random', 'balanced', t)
#     t = t - 0.0001
#
# fig = plt.figure()

# x = list(res.keys())
# y = list(res.values())
# #
# # x_new = np.linspace(min(x), max(x), 50)
# #
# # f = interpolate.interp1d(x, y, kind='quadratic')
# # y_smooth = f(x_new)
#
# plt.plot(x, y)
# plt.show()
# print(res)
# solution.Refine('entropy', 90, 'random', 'balanced', 0.00001)

# solution.Predict()
# solution.Show()
