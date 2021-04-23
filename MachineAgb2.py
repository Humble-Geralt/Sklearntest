from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from math import e
import matplotlib
import numpy as np

zhfont1 = matplotlib.font_manager.FontProperties(fname="SourceHanSansSC-Bold.otf") 

def cost_function(theta,X,Y,m):
    temp = 1/(1+e**np.dot(X,-theta))-Y
    return (1/(2*m))*np.dot(temp.transpose(),temp)

def gradient_function(theta,X,Y,m):
    temp = 1/(1+e**np.dot(X,-theta))-Y
    return (1/m)*np.dot(X.transpose(),temp)

def gradient_descent(X,Y,alpha,m):
    theta = np.array([1,1,1]).reshape(3,1)
    gradient = gradient_function(theta,X,Y,m)
    for i in range(5000):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta,X,Y,m)
    return theta

m=100
#定义学习率/步长为0.01
alpha=0.01

iris = load_iris()
iris_feature = iris.data
iris_label = iris.target

#截取前100个样本的特征和标签
X=iris_feature[:100,:2]
X_marixadd=np.ones((m,1))
X=np.hstack((X,X_marixadd))
Y=iris_label[:100].reshape(100,1)

optimal = gradient_descent(X,Y,alpha,m)
print(optimal)
print(gradient_function(optimal,X,Y,m))

fig=plt.figure(figsize=(14, 8))
plt.suptitle('对数几率回归模型', fontsize = 30,fontproperties=zhfont1)
ax = Axes3D(fig)
ax.scatter(X[:,0],X[:,1],Y)
x=np.linspace(0,10,20)
y=np.linspace(0,6,20)
X_area,Y_area=np.meshgrid(x,y)
Z=optimal[0]*X_area+optimal[1]*Y_area+optimal[2]
ax.plot_surface(X_area,Y_area,Z=1/(1 + e**-Z),color='r',alpha=0.5)
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
plt.show()
