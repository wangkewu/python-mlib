import numpy as np
import pandas as pd

# 这里直接引入sklearn里的数据集，iris鸢尾花
from sklearn.datasets import load_iris
#切分数据集维训练集和测试集
from sklearn.model_selection import train_test_split
#计算分类预测的准确率
from sklearn.metrics import accuracy_score

#数据加载和预处理
iris = load_iris()
# print(iris)
df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
#增加一列
df['class'] = iris.target
#改成具体映射的值
df['class'] = df['class'].map({0:iris.target_names[0], 1:iris.target_names[1], 2:iris.target_names[2]})
# print(df)
x = iris.data
y = iris.target.reshape(-1, 1)
# print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=36, stratify=y)
# print(x_train.shape, y_train.shape)
# print(y_train)
# print(df)


# print(x_test[0])
# print(x_test[0].reshape(1,-1).shape)
# print("--------------------")
# print(x_train.shape)
# print(x_test.shape)
# print("-------------------------------")
# print(np.sum(np.abs(x_train - x_test[0].reshape(1, -1)), axis=1))
#2核心算法实现
def l1_distance(a, b):
	#a 一个矩阵  b一个向量（）
	return np.sum(np.abs(a-b), axis=1)
def l2_distance(a, b):
	return np.sqrt(np.sum((a-b)**2, axis=1))

#3分类器实现
class KNN(object):
	#定义一个初始化方法 __init__ 是类的构造方法
	def __init__(self, n_nerighbors = 1, dist_func=l1_distance):
		self.n_nerighbors = n_nerighbors
		self.dist_func = dist_func

	#训练模型方法
	def fit(self, x, y):
		self.x_train = x
		self.y_train = y

	#模型预测方法
	def predict(self, x):

		#初始化预分类数组 x.shape[0]行 1列 初始化为0
		y_pred = np.zeros((x.shape[0], 1), dtype=self.y_train.dtype)

		#遍历输入的x数据点 取出每个数据点的序号和参数
		for i, x_test in enumerate(x):
			#step1 x_test 计算距离 所有训练数据计算距离
			distance = self.dist_func(self.x_train, x_test)
			#step2 得到的距离 按照有近到远排序 取出索引值
			nn_index = np.argsort(distance)
			#step3 选取其中最近的k个点 保存他们对应的分类类别
			nn_y = self.y_train[nn_index[:self.n_nerighbors]].ravel()
			#step4 统计类别出现平率最高的那个 赋值给y_pred[i]
			y_pred[i] = np.argmax(np.bincount(nn_y))
		return y_pred

#4 测试

knn = KNN(n_nerighbors = 3)
#训练
knn.fit(x_train, y_train)
#预测
y_pred = knn.predict(x_test)
# print(y_pred)
#求出预测准确率
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
# knn = KNN()
#训练
# knn.fit(x_train, y_train)
#保存结果
# result_list = []

#针对不同的参数选举 做预测
# for p in [1, 2]:
# 	knn.dist_func = l1_distance if p == 1 else l2_distance
# 	#考虑不同的k取值
# 	for k in range(1, 10, 2):
# 		knn.n_nerighbors = k
# 		y_pred = knn.predict(x_test)
# 		accuracy = accuracy_score(y_test, y_pred)	
# 		result_list.append([k, 'l1_distance' if p == 1 else 'l2_distance', accuracy])

# df = pd.DataFrame(result_list, columns = ['k', '距离函数', '预测准确度'])
# print(df)