import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
#warnings.filterwarnings('ignore')
#%matplotlib inline
from sklearn.metrics import roc_auc_score
## 数据降维处理的
from sklearn.model_selection import train_test_split  
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

train=pd.read_csv("train.csv")
print(train.shape)
print(train.columns)
print(train.info())
testA=pd.read_csv("testA.csv")

numerical_fea = list(train.select_dtypes(exclude=['object']).columns)
numerical_fea.remove('isDefault')
train[numerical_fea] = train[numerical_fea].fillna(train[numerical_fea].median())
testA[numerical_fea] = testA[numerical_fea].fillna(testA[numerical_fea].median())
#issueDate
#数据清洗
for data in [train]:
    data['issueDate'] = pd.to_datetime(data['issueDate'],format='%Y-%m-%d')
    data['grade'] = data['grade'].map({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7})
    data['employmentLength'] = data['employmentLength'].map({'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5,'6 years':6,'7 years':7,'8 years':8,'9 years':9,'10+ years':10,'< 1 year':0})
    data['subGrade'] = data['subGrade'].map({'E2':1,'D2':2,'D3':3,'A4':4,'C2':5,'A5':6,'C3':7,'B4':8,'B5':9,'E5':10,
        'D4':11,'B3':12,'B2':13,'D1':14,'E1':15,'C5':16,'C1':17,'A2':18,'A3':19,'B1':20,
        'E3':21,'F1':22,'C4':23,'A1':24,'D5':25,'F2':26,'E4':27,'F3':28,'G2':29,'F5':30,
        'G3':31,'G1':32,'F4':33,'G4':34,'G5':35})
    data['earliesCreditLine'] = data['earliesCreditLine'].apply(lambda s: int(s[-4:]))
  #  data['n15']=data['n8']*data['n10']
    
for data in [testA]:
    data['issueDate'] = pd.to_datetime(data['issueDate'],format='%Y-%m-%d')
    data['grade'] = data['grade'].map({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7})
    data['employmentLength'] = data['employmentLength'].map({'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5,'6 years':6,'7 years':7,'8 years':8,'9 years':9,'10+ years':10,'< 1 year':0})
    data['subGrade'] = data['subGrade'].map({'E2':1,'D2':2,'D3':3,'A4':4,'C2':5,'A5':6,'C3':7,'B4':8,'B5':9,'E5':10,
        'D4':11,'B3':12,'B2':13,'D1':14,'E1':15,'C5':16,'C1':17,'A2':18,'A3':19,'B1':20,
        'E3':21,'F1':22,'C4':23,'A1':24,'D5':25,'F2':26,'E4':27,'F3':28,'G2':29,'F5':30,
        'G3':31,'G1':32,'F4':33,'G4':34,'G5':35})
    data['earliesCreditLine'] = data['earliesCreditLine'].apply(lambda s: int(s[-4:]))

print("数据预处理完成!")  
sub=testA[['id']].copy()
sub['isDefault']=0
testA=testA.drop(['id','issueDate'],axis=1)
data_x=train.drop(['isDefault','id','issueDate'],axis=1)
data_y=train[['isDefault']].copy()

col=['grade','subGrade','employmentTitle','homeOwnership','verificationStatus','purpose','postCode','regionCode',
     'initialListStatus','applicationType','policyCode']
for i in data_x.columns:
    if i in col:
        data_x[i] = data_x[i].astype('str')
for i in testA.columns:
    if i in col:
        testA[i] = testA[i].astype('str')
        
model=CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            task_type="CPU",
            learning_rate=0.1,
            iterations=700,
            random_seed=2000,
            od_type="Iter",
            depth=7)

#k折交叉验证，此处k为5
answers = []
mean_score = 0
n_folds = 5
sk = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2019)
for train, test in sk.split(data_x, data_y):
    x_train = data_x.iloc[train]
    y_train = data_y.iloc[train]
    x_test = data_x.iloc[test]
    y_test = data_y.iloc[test]
    clf = model.fit(x_train,y_train, eval_set=(x_test,y_test),verbose=5,cat_features=col)
    yy_pred_valid=clf.predict(x_test)
    print('cat验证的auc:{}'.format(roc_auc_score(y_test, yy_pred_valid)))
    mean_score += roc_auc_score(y_test, yy_pred_valid) / n_folds
    y_pred_valid = clf.predict(testA,prediction_type='Probability')[:,-1]
    answers.append(y_pred_valid)

print('mean valAuc:{}'.format(mean_score))
cat_pre=sum(answers)/n_folds
sub['isDefault']=cat_pre
sub.to_csv('金融预测.csv',index=False) 
