from sklearn import svm
from matplotlib import pyplot as plt
import numpy as np
import scipy

TP=np.array([1,2,3,3,4,5,5,6,6,6,6,6])
FN=np.array([5,4,3,3,2,1,1,0,0,0,0,0])
FP=np.array([0,0,0,1,1,1,2,2,3,4,5,6])
TN=np.array([6,6,6,5,5,5,4,4,3,2,1,0])
P=np.array([])
R=np.array([])
TPR=np.array([])
FPR=np.array([])

for i in range(12):
    P=np.append(P,TP[i]/(TP[i]+FP[i]))
    R=np.append(R,TP[i]/(TP[i]+FN[i]))

    TPR=np.append(TPR,TP[i]/(TP[i]+FN[i]))
    FPR=np.append(FPR,FP[i]/(TN[i]+FP[i]))

plt.figure(1)
ax1 = plt.subplot(1,2,1)
plt.xlabel("Recall")
plt.ylabel("Preccision")
plt.plot(R,P,)
ax2 = plt.subplot(1,2,2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot(FPR,TPR)
ax1.set_title("P-R Pic")
ax2.set_title("ROC Pic")
plt.tight_layout()
plt.show()
