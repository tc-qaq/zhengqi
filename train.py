import sklearn.metrics
from sklearn.metrics import precision_recall_curve, roc_curve, auc,roc_auc_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import xgboost as xgb
import time

import scipy as sp
import numpy as np
from sklearn.cross_validation import train_test_split



def loadData():
    fr = open("zhengqi_train.txt",'r',encoding='utf-8')
    X = []
    Y = []
    index = 0
    for line in fr.readlines():
        index+=1
        if index == 1:
            continue        
        temp = line.strip().split('\t')
        x = temp[:-1]
        # remove v5, v9, v11, v17, v22, v28
        for i in range(len(x)):
            x[i] = float(x[i])        
        y = float(temp[-1])
        X.append(x)
        Y.append(y)
    return X,Y

def train(X, Y, model_dir):
    start_time = time.time()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1)
    xgb_train = xgb.DMatrix(x_train,y_train)
    xgb_test  = xgb.DMatrix(x_test,y_test)
    #调参步骤：max_depth(3-10) ->min_child_weight(2-12 ,step=2)->  
    #        -->gamma(0-0.5 step=0.1)  -> sample(0.6-1,step=0.1)
    #        -->alpha(1e-5-100,bi-cro)
    params = {
        'booster':'gbtree', 'silent':1 , 'eta': 0.02,#(0.01-0.2)        
        
        'max_depth':5, 'min_child_weight':0.8, 
        'gamma':0.1,'subsample':0.85, 'colsample_bytree':1, 
        'lambda':1, 'alpha':0, 
        
        'seed':1000, 'objective': 'reg:linear','eval_metric': 'rmse'
        }
    plst = list(params.items())
    num_rounds = 10000 # 迭代次数
    watchlist = [(xgb_train, 'train'),(xgb_test, 'val')]
    #训练模型并保存
    # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
    model = xgb.train(plst, xgb_train, num_rounds, watchlist)    
    print ("best best_ntree_limit",model.best_ntree_limit) 
    y_pred = model.predict(xgb_test,ntree_limit=model.best_ntree_limit)  
    #输出运行时长
    cost_time = time.time()-start_time
    print ("xgboost success!",'\n',"cost time:",cost_time,"(s)......")

    model.save_model(model_dir)

def test(model_dir):
    fr = open("zhengqi_test.txt",'r',encoding='utf-8')
    X = []
    index = 0
    for line in fr.readlines():
        index+=1
        if index == 1:
            continue        
        temp = line.strip().split('\t')
        x = temp[:-1]
        for i in range(len(x)):
            x[i] = float(x[i])        
        X.append(x)
    
    model = xgb.Booster()
    model.load_model(model_dir)
    dtest = xgb.DMatrix(X)
    answer = model.predict(dtest)
    fw = open('submit.txt','w')
    for item in answer:
        item = "%.6f"%item
        fw.write(item+'\n')
    fw.close()
    
X,Y = loadData()
train(X, Y, 'xgb_model')

test('xgb_model')