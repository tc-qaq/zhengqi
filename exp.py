import pandas as pd
import numpy as np
from sklearn import preprocessing
import math
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC,LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import xgboost as xgb



train=pd.read_csv('zhengqi_train.txt',sep='\t')
test=pd.read_csv('zhengqi_test.txt',sep='\t')
train_x=train.drop(['target'],axis=1)
all_data = pd.concat([train_x,test]) 
#remove v5... for consistently distribution
all_data.drop(['V5','V17','V28','V22','V11','V9'],axis=1,inplace=True)
# normalization
min_max_saclar = preprocessing.MinMaxScaler()
data_minmax = pd.DataFrame(min_max_saclar.fit_transform(all_data), columns=all_data.columns)
data_minmax['V0'] = data_minmax['V0'].apply(lambda x:math.exp(x))
data_minmax['V1'] = data_minmax['V1'].apply(lambda x:math.exp(x))
data_minmax['V6'] = data_minmax['V6'].apply(lambda x:math.exp(x))
data_minmax['V30'] = np.log1p(data_minmax['V30'])

X_scaled = pd.DataFrame(preprocessing.scale(data_minmax),columns = data_minmax.columns)
train_x = X_scaled.ix[0:len(train)-1]
test = X_scaled.ix[len(train):]
Y=train['target']
# feature selsection, remove var <0.85 -> change is small ->meanless
threshold = 0.85                  
vt = VarianceThreshold().fit(train_x)
# Find feature names
feat_var_threshold = train_x.columns[vt.variances_ > threshold * (1-threshold)]
train_x = train_x[feat_var_threshold]
test = test[feat_var_threshold]

X_scored = SelectKBest(score_func=f_regression, k='all').fit(train_x, Y)
feature_scoring = pd.DataFrame({
        'feature': train_x.columns,
        'score': X_scored.scores_
    })
head_feature_num = 18
feat_scored_headnum = feature_scoring.sort_values('score', ascending=False).head(head_feature_num)['feature']
train_x_head = train_x[train_x.columns[train_x.columns.isin(feat_scored_headnum)]]
test_x_head  = test[train_x.columns[train_x.columns.isin(feat_scored_headnum)]]

X_scaled = pd.DataFrame(preprocessing.scale(train_x),columns = train_x.columns)

n_folds = 10

def rmsle_cv(model,train_x_head=train_x_head):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_x_head)
    rmse= -cross_val_score(model, train_x_head, Y, scoring="neg_mean_squared_error", cv = kf)
    return(rmse)
    
svr = make_pipeline( SVR(kernel='linear')) 
line = make_pipeline( LinearRegression())
lasso = make_pipeline( Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline( ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR1 = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
KRR2 = KernelRidge(alpha=1.5, kernel='linear', degree=2, coef0=2.5)
model_xgb = xgb.XGBRegressor(booster='gbtree',colsample_bytree=0.8, gamma=0.1, learning_rate=0.02, 
                             max_depth=5, n_estimators=500,min_child_weight=0.8, reg_alpha=0, 
                             reg_lambda=1, subsample=0.8, silent=1, random_state =42, nthread = 2)


score = rmsle_cv(svr)
print("\nSVR 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
svr.fit(train_x_head,Y)
score = rmsle_cv(line)
print("\nLine 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(lasso)
print("\nLasso 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)
print("ElasticNet 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(KRR2)
print("Kernel Ridge2 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
KRR2.fit(train_x_head,Y)
head_feature_num = 18
feat_scored_headnum = feature_scoring.sort_values('score', ascending=False).head(head_feature_num)['feature']
train_x_head2 = train_x[train_x.columns[train_x.columns.isin(feat_scored_headnum)]]
X_scaled = pd.DataFrame(preprocessing.scale(train_x),columns = train_x.columns)
score = rmsle_cv(KRR1,train_x_head2)
print("Kernel Ridge1 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
head_feature_num = 22
feat_scored_headnum = feature_scoring.sort_values('score', ascending=False).head(head_feature_num)['feature']
train_x_head3 = train_x[train_x.columns[train_x.columns.isin(feat_scored_headnum)]]
X_scaled = pd.DataFrame(preprocessing.scale(train_x),columns = train_x.columns)
score = rmsle_cv(model_xgb,train_x_head3)
print("Xgboost 得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_xgb.fit(train_x_head,Y)

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # 遍历所有模型，你和数据
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        for model in self.models_:
            model.fit(X, y)
        return self
    
    # 预估，并对预估结果值做average
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        #return 0.85*predictions[:,0]+0.15*predictions[:,1]
        return 0.7*predictions[:,0]+0.15*predictions[:,1]+0.15*predictions[:,2]
        #return np.mean(predictions, axis=1)   
        
#averaged_models = AveragingModels(models = (lasso,KRR))    
averaged_models = AveragingModels(models = (svr,KRR2,model_xgb))

score = rmsle_cv(averaged_models)
print(" 对基模型集成后的得分: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

models = averaged_models.fit(train_x_head,Y)
pres = averaged_models.predict(test_x_head)
fw = open('submit_avg.txt','w')
for item in pres:
    item = "%.6f"%item
    fw.write(item+'\n')
fw.close()

