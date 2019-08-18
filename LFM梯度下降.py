import numpy as np
import pandas as pd

#数据准备
R = np.array([[4,0,2,0,1],
			  [0,2,3,0,0],
			  [1,0,2,4,0],
			  [5,0,0,3,1],
			  [0,0,1,5,1],
			  [0,3,2,4,1]])

print(R.shape[0])
# print(len(R[0]))
# print(len(R[1]))
# print(R.shape[1])
#算法实现

'''
输入参数
R： M*N的评分矩阵
K: 特征向量个数
max_iter:最大迭代数
alpha:步长
lamda:正则化系数

输出
分解之后的P Q
P:初始化用户特征矩阵 M*K
Q:初始化物品特征举证N*K
Q做一个转置Qt
'''
K = 2
max_iter = 5000
alpha = 0.0002
lamda = 0.004

#核心算法
def LFM_grad_desc(R, max_iter, lamda, K=2,  alpha=0.0001):
	#基本参数定义
	M = len(R)
	N = len(R[0])
	#P Q初始值 随机生成
	P = np.random.rand(M, K)
	Q = np.random.rand(N, K)
	Q = Q.T #转置矩阵 K * N

	#开始迭代
	for step in range(max_iter):
		#对所有的用户u 物品i做遍历 对应的特征向量 Pu Qi梯度下降
		for u in range(M):
			for i in range(N):
				#对于每一个大于0的评分 求出预测评分误差
				if R[u][i] > 0:
					#用户行特征与物品特征求出一个数
					eui = np.dot(P[u,:], Q[:, i]) - R[u][i]

					#带入公式 按照梯度下降算法更新当前的Pu Qi
					for k in range(K):
						P[u][k] = P[u][k] - alpha *(2 * eui * Q[k][i]+ 2 * lamda * P[u][k])
						Q[k][i] = Q[k][i] - alpha *(2 * eui * P[u][k]+ 2 * lamda * Q[k][i])
		#u 	i 遍历完成 所有特征向量更新完成 可以得到P Q 可以计算预测评分矩阵
		predR = np.dot(P, Q)

		#计算当前损失函数
		cost = 0
		for u in range(M):
			for i in range(N):
				if R[u][i] > 0:
					cost += (np.dot(P[u, :], Q[:, i]) - R[u][i])**2
					#加上正则化项
					for k in range(K):
						cost += lamda * (P[u][k]**2 + Q[k][i] ** 2)
		if cost < 0.0001:
			break
	return P, Q.T, cost


#测试
P, Q , cost = LFM_grad_desc(R, max_iter, lamda, K, alpha)

print(P)
print(Q)
print(cost)
predR = P.dot(Q.T)
print(predR)