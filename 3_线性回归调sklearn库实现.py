import numpy as np
import matplotlib.pyplot as plt
points = np.genfromtxt('data.csv', delimiter=',')
# print(points[0][0])
# print(points)
x = points[:, 0]
y = points[:, 1]
# plt.scatter(x,y)
# plt.show()

#损失函数是系数函数 另外还要传入数据的x,y
def compute_cost(w, b, points):
	total_cost = 0
	M = len(points)

	#逐点计算放平损失误差 然后求平均数
	for i in range(M):
		x = points[i, 0]
		y = points[i, 1]
		total_cost += (y -w*x -b) ** 2

	return total_cost/M

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

x_new = x.reshape(-1, 1)
y_new = y.reshape(-1, 1)
lr.fit(x_new, y_new)


w = lr.coef_[0][0]
b = lr.intercept_[0]


print("w is :", w)
print("b is: ", b)

cost = compute_cost(w, b, points)
print("cost is:", cost)

plt.scatter(x, y)

pred_y = w*x + b

plt.plot(x, pred_y, c='red')
plt.show()