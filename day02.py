import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from bz2 import __author__
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#sess = tf.InteractiveSession()

# I_matrix = tf.eye(5)
# print(I_matrix.eval())
#
# X = tf.Variable(tf.eye(5))
#
# X.initializer.run()   #初始化X
#
# A = tf.Variable(tf.random_normal([5,5],seed=0))
# A.initializer.run()
#
# product = tf.matmul(A,X)
#
# print(product.eval())
#
# b = tf.Variable(tf.random_uniform([5,5],0,2,dtype=tf.int32,seed=0))
# b.initializer.run()
# print(b.eval())
# #cast函数将b的数据格式转化成dtype数据类型,相当于一个类型转换函数
# b_new = tf.cast(b,dtype=tf.float32)
#
# print('shape product:',product.shape)
#
# print('shape b_new',b_new.shape)
#
# t_sum = tf.add(product,b_new)
#
#
# print(b_new.eval())
# print('A的type：' + str(A.dtype))
# print('b_new的type：' + str(b_new.dtype))
#
#
#
#
#
# #print(X.eval())
# #初始化并输出一个变量还有的另外一种方式为
# #令变量为X
# #sess = tf.compat.v1.Session()
# #sess.run(tf.global_variables_initializer)
# #sess.run(X)
# #
#
#
# #print(sess.run(X))

#A = tf.Variable(tf.random_uniform([2,3],0,10,dtype=tf.int32,seed=0))
#A.initializer.run()


#B = tf.Variable(tf.random_uniform([3,2],0,2,dtype=tf.int32,seed=0))
#B.initializer.run()


#mul_result = tf.matmul(A,B)

#add_result = tf.add(A,B)

#print('mul_result:')
#print(mul_result.eval())

#print('add_result:')
#print(add_result.eval())

#X = 2 * np.random.rand(100,1)

#y = 4 + 3 * X + np.random.randn(100,1)

#X = 2 * np.random.randn(100,1)
#X_b = np.c_[np.ones((100,1)),X]
#print(X_b)
#
# a = np.array([[1,2,3],[7,8,9]])
#
# b = np.array([[4,5,6],[1,2,3]])
#
# print(a)
# print(b)
# #按列将b放到a的右边
# c = np.c_[a,b]
# print(c)



#sess.close()
seed = np.random.seed(100)

# 构造一个100行1列到矩阵。矩阵数值生成用rand，得到到数字是0-1到均匀分布到小数。
X = 2 * np.random.rand(100, 1)  # 最终得到到是0-2均匀分布到小数组成到100行1列到矩阵。这一步构建列X1(训练集数据)
# 构建y和x的关系。 np.random.randn(100,1)是构建的符合高斯分布（正态分布）的100行一列的随机数。相当于给每个y增加列一个波动值。
y = 4 + 3 * X + np.random.randn(100, 1)

# 将两个矩阵组合成一个矩阵。得到的X_b是100行2列的矩阵。其中第一列全都是1.
X_b = np.c_[np.ones((100, 1)), X]

# 解析解求theta到最优解
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# print(theta_best)
# 生成两个新的数据点,得到的是两个x1的值

print('theta_best:')
print(theta_best)
X_new = np.array([[0], [2]])

# 填充x0的值，两个1
X_new_b = np.c_[(np.ones((2, 1))), X_new]

print(X_new_b)

# 用求得的theata和构建的预测点X_new_b相乘，得到yhat
y_predice = X_new_b.dot(theta_best)
print(y_predice)

# 画出预测函数的图像，r-表示为用红色的线
plt.plot(X_new, y_predice, 'r-')

# 画出已知数据X和掺杂了误差的y，用蓝色的点表示
plt.plot(X, y, 'b.')

# 建立坐标轴
plt.axis([0, 2, 0, 15])

plt.show()
