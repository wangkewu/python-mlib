import numpy as np
import matplotlib.pyplot as plt
points = np.genfromtxt('data.csv', delimiter=',')
# print(points[0][0])
# print(points)
x = points[:, 0]
y = points[:, 1]

#plt.scatter(x,y)
#plt.show()
# str = [[1 2],[3 4],[4 5]]
# print(str[0])

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

#3定义模型的超参数

alpha = 0.0001
initial_w = 0
initial_b = 0
num_ite = 10

#定义核心梯度下降算法函数


def grad_desc(points, initial_w, initial_b, alpha, num_ite):
	w = initial_w
	b = initial_b

	#定义一个list 保存所有的损失函数值 用来显示下降的过程
	cost_list = []

	for i in range(num_ite):
		cost_list.append(compute_cost(w, b, points))
		w , b = step_grad_desc(w, b, alpha, points)

	return [w, b, cost_list]

def step_grad_desc(current_w, current_b, alpha, points):
	sum_grad_w = 0
	sum_grad_b = 0
	M = len(points)
	#对每个点代入公式求和
	for i in range(M):
		x = points[i, 0]
		y = points[i, 1]
		sum_grad_w += (current_w * x + current_b - y)*x
		sum_grad_b += current_w * x +current_b - y

	#利用公式求当前梯度
	grade_w = 2/M*sum_grad_w
	grade_b = 2/M*sum_grad_b

	#梯度下降 更新当前的w b
	updated_w = current_w - alpha * grade_w
	updated_b = current_b - alpha * grade_b

	return updated_w, updated_b

#测试 运行梯度下降算法 计算最优的w 和b
'''alpha = 0.0001
initial_w = 0
initial_b = 0
num_ite = 10'''
w, b, cost_list = grad_desc(points, initial_w, initial_b, alpha, num_ite)

print("w is:", w)
print("b is:", b)
cost = compute_cost(w, b, points)
print("cost is:", cost)
#下降曲线 图
# plt.plot(cost_list)

# plt.show()

plt.scatter(x, y)

pred_y = w*x + b

plt.plot(x, pred_y, c='red')
plt.show()