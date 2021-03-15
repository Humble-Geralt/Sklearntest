from sklearn.metrics import *
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize,StandardScaler
from sklearn.multiclass import *
from matplotlib import pyplot as plt
from itertools import cycle
import matplotlib
import numpy as np

zhfont1 = matplotlib.font_manager.FontProperties(fname="SourceHanSansSC-Bold.otf") 

def draw_data(X_train,names,title):
    fig = plt.figure('Iris Data', figsize=(15,15))
    plt.suptitle('鸢尾花数据集', fontsize = 30,fontproperties=zhfont1)
    for i in range(4):
        for j in range(4):
            plt.subplot(4,4,4*i+(j+1))
            if i==j:
                plt.text(0.3,0.5,names[i],fontsize = 25,fontproperties=zhfont1)
            else:
                #两两组合绘制带有种类的散点图，RGB对应name[i],name[j]以及种类
                plt.scatter(np.array(X_train)[:,j], np.array(X_train)[:,i], c=np.array(X_train)[:,3], cmap = 'brg')
            if i==0:
                plt.title(names[j], fontsize=20,fontproperties=zhfont1) 
            if j == 0:
                plt.ylabel(names[i],fontsize=20,fontproperties=zhfont1)
    plt.tight_layout(rect=[0,0,1,0.9])
    plt.savefig(title+'data-view.png')
    plt.show()
    return 0

def num_get(y_score,y_test,classes):
    Precision=dict()
    Recall=dict()
    TPR=dict()
    FPR=dict()
    for i in range(classes):
        Precision[i],Recall[i],_ = precision_recall_curve(y_test[:,i],y_score[:,i])
        FPR[i],TPR[i],_ = roc_curve(y_test[:,i], y_score[:,i])
    return Precision,Recall,TPR,FPR

def draw_line(Precision,Recall,TPR,FPR,classes,title): 
    colors = cycle(['navy', 'turquoise', 'darkorange', ])
    plt.figure(figsize=(14, 8))
    plt.suptitle(title+' 的P-R曲线和ROC曲线', fontsize = 30,fontproperties=zhfont1)
    ax1=plt.subplot(1,2,1)
    plt.xlabel("Recall")
    plt.ylabel("Preccision")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    for i, color in zip(range(classes), colors):
        l, = plt.plot(Recall[i], Precision[i], color=color)
    ax2=plt.subplot(1,2,2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    for i, color in zip(range(classes), colors):
        print(color)
        l, = plt.plot(FPR[i], TPR[i], color=color)
    ax1.set_title("P-R Pic")
    ax2.set_title("ROC Pic")
    plt.savefig(title+'PR-ROC.png')
    plt.show()
    return 0

iris = load_iris()

iris_feature = iris.data
iris_label = iris.target
iris_label_res = label_binarize(iris_label,classes=[0,1,2])
n_classes = iris_label_res.shape[1]

random_state = np.random.RandomState(0)
n_samples, n_features = iris_feature.shape
print(random_state.randn(n_samples, 1 * n_features))
iris_feature_rand = np.c_[iris_feature, random_state.randn(n_samples, 200 * n_features)]
X_trainS, X_testS, y_trainS, y_testS = train_test_split(iris_feature_rand, iris_label_res, test_size=0.5, random_state=random_state)
X_trainL, X_testL, y_trainL, y_testL = train_test_split(iris_feature_rand, iris_label, test_size=0.5, random_state=random_state)
y_testL = label_binarize(y_testL,classes=[0,1,2])

title = '鸢尾花数据集'
names = ['花萼长度','花萼宽度','花瓣长度','花瓣宽度','品种']
draw_data(X_trainS,names,title)

#建立支持向量机
clf1 = OneVsRestClassifier(svm.LinearSVC(random_state=random_state))
#训练
clf1.fit(X_trainS,y_trainS)

clf2 = OneVsRestClassifier(LogisticRegression(random_state=random_state))
clf2.fit(X_trainL,y_trainL)
#得到预测的标签
y_scoreS=clf1.decision_function(X_testS)
y_scoreL=clf2.decision_function(X_testL)

scaler = StandardScaler().fit(y_scoreL)
y_scoreL = scaler.transform(y_scoreL)#标准化后的数据
print(y_testS,y_testL)

SVMP,SVMR,SVMTPR,SVMFPR=num_get(y_scoreS,y_testS,n_classes)
LRP,LRR,LRTPR,LRFPR=num_get(y_scoreL,y_testL,n_classes)

draw_line(SVMP,SVMR,SVMTPR,SVMFPR,n_classes,'SVM')
draw_line(LRP,LRR,LRTPR,LRFPR,n_classes,'LR')

