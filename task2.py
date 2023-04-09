# utf-8 encoded
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import random
import matplotlib.pyplot as plt

csv = pd.read_csv('height_data.csv')
data = csv['height'].to_numpy().reshape(-1, 1)


def gaussian(x, miu, cov):
    norm = multivariate_normal(mean=miu, cov=cov)
    return norm.pdf(x)


def em_single(x, K=2, iter_number=100):
    N = len(x)
    # alpha = np.random.rand(K, 1)
    # miu = np.random.rand(K, 1) * 20 + 155
    # cov = np.random.rand(K, 1) * 10
    # omega = np.ones((N, K)) / K
    alpha = [0.5, 0.5]
    miu = [160, 170]
    cov = [10, 10]
    omega = np.ones((N, K))

    for i in range(iter_number):
        # E-step
        temp = np.zeros((N, K))
        for k in range(K):
            temp[:, k] = alpha[k] * gaussian(x, miu[k], cov[k])
        sumT = np.sum(temp, axis=1)
        omega = temp / sumT[:, None]

        # M-step
        for k in range(K):
            omega_sum = np.sum(omega[:, k])
            alpha[k] = omega_sum / N
            miu[k] = np.dot(omega[:, k], x) / omega_sum
            cov[k] = np.dot(omega[:, k], np.square(x - miu[k])) / omega_sum

    return alpha, miu, cov, omega


alpha, miu, cov, omega = em_single(data)
print('EM算法均值:' + str(miu[0][0]) + ', ' + str(miu[1][0]) + '\n方差：' + str(cov[0][0]) + ', ' + str(cov[1][0]))

miuF = 164
covF = 3*3
miuM = 176
covM = 5*5
print('实际均值：%f, %f, 实际方差：%f, %f' % (float(miuF), float(miuM), covF, covM))

print('均值偏差：%f, %f' % (miuF - min(miu), miuM - max(miu)))

i = 0
for j, line in enumerate(omega):
    if j == 500:
        res1 = i
    if line[0] > line[1]:
        i += 1
print('估计男性人数：', max(i, len(data) - i))
print('估计女性人数：', min(i, len(data) - i))

print(res1)
print('预测准确率：', (2000 - 2 * min(res1, 500-res1)) / 2000)
