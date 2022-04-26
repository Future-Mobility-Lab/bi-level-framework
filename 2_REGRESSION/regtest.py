#!/usr/bin/env python
# coding: utf-8

# In[1]:


from MLimports2 import *


# In[2]:


from sklearn.datasets import make_regression


# In[29]:


X,Y = make_regression(n_samples=200, n_features=20, n_informative=5)


# In[30]:


def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = y_true[np.nonzero(y_true)] # avoid incidents with zero duration
    y_pred = y_pred[np.nonzero(y_true)]
    return abs(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

def smape(A, F):
    return 100/len(A) * np.sum( np.abs(F - A) / ((np.abs(A) + np.abs(F))/2.0) )

#second implementation
# def smape(A, F):
#     A = np.abs(np.mean(A)-F)
#     B = np.abs(np.mean(A)) + np.abs(F)
#     return 100.0*np.sum(A/B)/len(A)

from scipy import stats
def r2abs(yt,yp):
    A = np.sum(np.power(yp - yt, 2))
    B = np.sum(np.power(yt - np.mean(yt), 2))
    return abs(1-A/B)

def r2opt(yt,yp):
    A = np.sum(np.power(yp - yt, 2))
    B = np.sum(np.power(yt - np.mean(yt), 2))
    return A/B

def r2(yt,yp):
    A = np.sum(np.power(yp - yt, 2))
    B = np.sum(np.power(yt - np.mean(yt), 2))
    return 1-A/B

def mse(yt,yp):
    return mean_squared_error(yt,yp)

def rmse(yt,yp):
    return np.sqrt(mean_squared_error(yt,yp))


#
def csmape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = y_true[np.nonzero(y_true)] # avoid incidents with zero duration
    y_pred = y_pred[np.nonzero(y_true)]
    CS = (np.abs(y_true - y_pred) > (y_true/6))
    y_pred[CS] -= y_true[CS]/6
    y_pred[~CS] = 0 
    return np.mean(CS*np.abs((y_true - y_pred) / y_true)) * 100


# In[31]:


from sklearn.neighbors import KNeighborsRegressor

OPTIMIZER = []
OPTIMIZER.append({'name':'XGB','instance':XGBRegressor,'param':{
            'learning_rate' : np.linspace(0.0001,0.5,2500), 
            'n_estimators' : range(20,200,1),
            'max_depth':range(1,15,1), ##MAR: start at 3 
            'subsample':[0.6,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
        }})

# OPTIMIZER.append({'name':'XGB2','instance':XGBRegressor,'param':{
#             'learning_rate' : np.linspace(0,0.5,50), 
#             'n_estimators' : range(20,200,1),
#             'max_depth':range(1,15,1), ##MAR: start at 3 
#             'subsample':[0.6,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
#         }})
# OPTIMIZER.append({'name':'KNN','instance':KNeighborsRegressor,'param':{
#             'n_neighbors' : range(2,100), 
#             'weights' : ['uniform','distance']
#         }})

# OPTIMIZER.append({'name':'LGBM','instance':lgb.LGBMRegressor,'param':{
#             'learning_rate' : np.linspace(0,0.5,250), 
#             'n_estimators' : range(20,200,1),
#             'num_leaves' : range(10,100,1),
#             'max_depth':range(1,15,1), ##MAR: start at 3 
#             'subsample':[0.6,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
#         }})

# OPTIMIZER.append({'name':'ELM','instance':ELMRegressor,'param':{
#             'n_hidden' : range(2,100), 
#             'alpha' : np.linspace(0,1,20),
#             'rbf_width':np.linspace(0,1,20)
#         }})

OPTIMIZER.append({'name':'GBDT','instance': GradientBoostingRegressor, 'param':{
    'learning_rate' : np.linspace(0.0001,0.5,2500), 
    'n_estimators' : range(20,200,1),
    'max_depth':range(3,15,1), ##MAR: start at 3 
    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
}})

OPTIMIZER.append({'name':'LR','instance': Ridge, 'param':{
    'alpha':np.linspace(0,1,100)
}})

# OPTIMIZER.append({'name':'RF','instance': RandomForestRegressor, 'param':{
#     'n_estimators' : range(20,200,1),
#     'max_depth':range(3,15,1), ##MAR: start at 3 
# }})


# In[32]:


METRIC = []
# METRIC.append({'name':'RMSE','instance':rmse})
METRIC.append({'name':'MAPE','instance':mape})
# METRIC.append({'name':'R2','instance':r2opt})
# METRIC.append({'name':'CSMAPE','instance':csmape})


# In[33]:


def EVAL_LOOCV(REG,MET):
    global RESULTS,X,Y
    from sklearn.model_selection import KFold
    import multiprocessing
    
    
    search = RandomizedSearchCV(estimator = REG['instance'](),
                       param_distributions=REG['param'],
                       n_iter=120,
                       scoring=make_scorer(MET['instance'], greater_is_better=False),
                       n_jobs=multiprocessing.cpu_count(), ## MAR: used to be 1
                       cv = 5,
                       verbose = 1
                           )
    search.fit(X, Y)
    reg = REG['instance'](**search.best_params_)
    
    print(REG['name'],MET['name'],search.best_params_)
    
#     FOLDS=10
    ATTEMPTS=1
    
    from sklearn.model_selection import LeaveOneOut
    kf = LeaveOneOut() #574/5 = 114
    kf.get_n_splits(X)
    
#     kf = KFold(n_splits=FOLDS)
#     kf.get_n_splits(X)

    SCOREMSE=[]
    SCORER2=[]
    SCORER2ABS=[]
    SCORESMAPE=[]
    SCOREMAPE=[]
    SCORECSMAPE=[]
    
    import tqdm
    for A in range(ATTEMPTS):
        PRED = []
        
        for train_index, test_index in tqdm.tqdm(kf.split(X)):
            Xtr, Xte = X.iloc[train_index], X.iloc[test_index]
            Ytr, Yte = Y[train_index], Y[test_index]
#             evaluation = [( Xtr,Ytr ), ( Xte, Yte)]
            #9 folds vs 1 fold

#             if log:
#                 Ytr=np.log1p(Ytr)

            reg.fit(Xtr,Ytr)
        
            pred111 = reg.predict(Xte)
#             print(pred111)
            PRED.append(pred111[0])

    #             if log:
    #                 pred=np.expm1(pred)
        PRED = np.array(PRED).ravel()
        SCOREMSE.append(mse(Y, PRED))
        SCOREMAPE.append(mape(Y, PRED))
        SCORER2.append(r2(Y,PRED))
        SCORER2ABS.append(r2abs(Y,PRED))
        SCORESMAPE.append(smape(Y,PRED))
        SCORECSMAPE.append(csmape(Y,PRED))

    RESULTS.append({'Optimizer': REG['name'],
      'Training': MET['name'],
      'mape': np.array(SCOREMAPE).mean(),
      'smape': np.array(SCORESMAPE).mean(), 'csmape': np.array(SCORECSMAPE).mean(),
      'r2': np.array(SCORER2).mean(), 'r2abs': np.array(SCORER2ABS).mean(),
      'mse': np.array(SCOREMSE).mean(),'folding':'LOO'})
    
def EVAL(REG,MET):
    global RESULTS,X,Y
    from sklearn.model_selection import KFold
    import multiprocessing
    
    FOLDS=10
    ATTEMPTS=1
    kf = KFold(n_splits=FOLDS,shuffle=True)
    kf.get_n_splits(X)

    SCOREMSE=[]
    SCORER2=[]
    SCORESMAPE=[]
    SCOREMAPE=[]
    import tqdm
    for A in range(ATTEMPTS):
        for train_index, test_index in tqdm.tqdm(kf.split(X)):
            Xtr, Xte = X.iloc[train_index], X.iloc[test_index]
            Ytr, Yte = Y[train_index], Y[test_index]
            evaluation = [( Xtr,Ytr ), ( Xte, Yte)]
            #9 folds vs 1 fold

            if log:
                Ytr=np.log1p(Ytr)

            search = RandomizedSearchCV(estimator = REG['instance'](),
                               param_distributions=REG['param'],
                               n_iter=50,
                               scoring=make_scorer(MET['instance'], greater_is_better=(MET['name']=='R2')),
                               n_jobs=multiprocessing.cpu_count(), ## MAR: used to be 1
                               cv = 5,
                               verbose = 1
                                   )
            search.fit(Xtr, Ytr)

            pred = search.predict(Xte)

            if log:
                pred=np.expm1(pred)

            SCOREMSE.append(mse(Yte, pred))
            SCOREMAPE.append(mape(Yte, pred))
            SCORER2.append(r2(Yte,pred))
            SCORESMAPE.append(smape(Yte,pred))

    RESULTS.append({'Optimizer': REG['name'],
      'Training': MET['name'],
      'mape': np.array(SCOREMAPE).mean(),
      'smape': np.array(SCORESMAPE).mean(),
      'r2': np.array(SCORER2).mean(),
      'mse': np.array(SCOREMSE).mean(),'folding':'KF10'})

    
def EVAL2(REG,MET,setup='AlltoAll',TH=45):
    global RESULTS,X,Y
    from sklearn.model_selection import KFold
    import multiprocessing
    
    
#     X_A, Y_A = getBFS(part=[5,TH])
#     X_B, Y_B = getBFS(part=[TH,-1])
    
    
    search = RandomizedSearchCV(estimator = REG['instance'](),
                       param_distributions=REG['param'],
                       n_iter=70,
                       scoring=make_scorer(MET['instance'], greater_is_better=False),
                       n_jobs=multiprocessing.cpu_count(), ## MAR: used to be 1
                       cv = 10,
                       verbose = 1
                           )
    search.fit(X, Y)
    reg = REG['instance'](**search.best_params_)
    
    
    
    FOLDS=10
    ATTEMPTS=2
    
    SCOREMSE=[]
    SCORER2=[]
    SCORER2ABS=[]
    SCORESMAPE=[]
    SCOREMAPE=[]
    SCORECSMAPE=[]
    
    import tqdm
    
    
    if setup=='AlltoA':
        for A in range(ATTEMPTS):
            
            kf = KFold(n_splits=FOLDS,shuffle=True)
    
            if setup=='AlltoAll':
                kf.get_n_splits(X)

            if setup=='AtoB':
                kf.get_n_splits(X_A)

            if setup=='AtoA':
                kf.get_n_splits(X_A)

            if setup=='BtoA':
                print('split BtoA')
                kf.get_n_splits(X_B)

            if setup=='AlltoB':
                print('split AlltoB')
                kf.get_n_splits(X_B)

            if setup=='AlltoA':
                print('split AlltoA')
                kf.get_n_splits(X_A)
            
            for train_index, test_index in tqdm.tqdm(kf.split(X_A)):
                
                Xtr, Xte = pd.concat([X_B,X_A.iloc[train_index]], axis=0), X_A.iloc[test_index]
                Ytr, Yte = np.concatenate([np.array(Y_B), np.array(Y_A)[train_index]],axis=0), Y_A[test_index]
                
                reg.fit(Xtr, Ytr)

                pred = reg.predict(Xte)
                
                PRED = np.array(pred).ravel()
                SCOREMSE.append(mse(Yte, PRED))
                SCOREMAPE.append(mape(Yte, PRED))
                SCORER2.append(r2(Yte,PRED))
                SCORER2ABS.append(r2abs(Yte,PRED))
                SCORESMAPE.append(smape(Yte,PRED))
                SCORECSMAPE.append(csmape(Yte,PRED))   
                
    if setup=='AlltoB':
        for A in range(ATTEMPTS):
            
            kf = KFold(n_splits=FOLDS,shuffle=True)
    
            if setup=='AlltoAll':
                kf.get_n_splits(X)

            if setup=='AtoB':
                kf.get_n_splits(X_A)

            if setup=='AtoA':
                kf.get_n_splits(X_A)

            if setup=='BtoA':
                print('split BtoA')
                kf.get_n_splits(X_B)

            if setup=='AlltoB':
                print('split AlltoB')
                kf.get_n_splits(X_B)

            if setup=='AlltoA':
                print('split AlltoA')
                kf.get_n_splits(X_A)
                
                
            for train_index, test_index in tqdm.tqdm(kf.split(X_B)):
                
                Xtr, Xte = pd.concat([X_A,X_B.iloc[train_index]], axis=0), X_B.iloc[test_index]
                Ytr, Yte = np.concatenate([np.array(Y_A), np.array(Y_B)[train_index]],axis=0), Y_B[test_index]
                
                reg.fit(Xtr, Ytr)

                pred = reg.predict(Xte)
                
                PRED = np.array(pred).ravel()
                SCOREMSE.append(mse(Yte, PRED))
                SCOREMAPE.append(mape(Yte, PRED))
                SCORER2.append(r2(Yte,PRED))
                SCORER2ABS.append(r2abs(Yte,PRED))
                SCORESMAPE.append(smape(Yte,PRED))
                SCORECSMAPE.append(csmape(Yte,PRED))
                
    if setup=='AtoA':
        for A in range(ATTEMPTS):
            
            kf = KFold(n_splits=FOLDS,shuffle=True)
    
            if setup=='AlltoAll':
                kf.get_n_splits(X)

            if setup=='AtoB':
                kf.get_n_splits(X_A)

            if setup=='AtoA':
                kf.get_n_splits(X_A)

            if setup=='BtoA':
                print('split BtoA')
                kf.get_n_splits(X_B)

            if setup=='AlltoB':
                print('split AlltoB')
                kf.get_n_splits(X_B)

            if setup=='AlltoA':
                print('split AlltoA')
                kf.get_n_splits(X_A)
                
            for train_index, test_index in tqdm.tqdm(kf.split(X_A)):
                Xtr, Xte = X_A.iloc[train_index], X_A.iloc[test_index]
                Ytr, Yte = Y_A[train_index], Y_A[test_index]

                reg.fit(Xtr, Ytr)

                pred = reg.predict(Xte)

                PRED = np.array(pred).ravel()
                SCOREMSE.append(mse(Yte, PRED))
                SCOREMAPE.append(mape(Yte, PRED))
                SCORER2.append(r2(Yte,PRED))
                SCORER2ABS.append(r2abs(Yte,PRED))
                SCORESMAPE.append(smape(Yte,PRED))
                SCORECSMAPE.append(csmape(Yte,PRED))
                
                
    if setup=='AtoB':
        for A in range(ATTEMPTS):
            
            kf = KFold(n_splits=FOLDS,shuffle=True)
    
            if setup=='AlltoAll':
                kf.get_n_splits(X)

            if setup=='AtoB':
                kf.get_n_splits(X_A)

            if setup=='AtoA':
                kf.get_n_splits(X_A)

            if setup=='BtoA':
                print('split BtoA')
                kf.get_n_splits(X_B)

            if setup=='AlltoB':
                print('split AlltoB')
                kf.get_n_splits(X_B)

            if setup=='AlltoA':
                print('split AlltoA')
                kf.get_n_splits(X_A)
            
            for train_index, test_index in tqdm.tqdm(kf.split(X_A)):
                Xtr, Xte = X_A.iloc[train_index], X_B
                Ytr, Yte = Y_A[train_index], Y_B

                reg.fit(Xtr, Ytr)

                pred = reg.predict(Xte)

                PRED = np.array(pred).ravel()
                SCOREMSE.append(mse(Yte, PRED))
                SCOREMAPE.append(mape(Yte, PRED))
                SCORER2.append(r2(Yte,PRED))
                SCORER2ABS.append(r2abs(Yte,PRED))
                SCORESMAPE.append(smape(Yte,PRED))
                SCORECSMAPE.append(csmape(Yte,PRED))
                
    if setup=='BtoA':
        print('folding BtoA')
        for A in range(ATTEMPTS):
            
            
            kf = KFold(n_splits=FOLDS,shuffle=True)
    
            if setup=='AlltoAll':
                kf.get_n_splits(X)

            if setup=='AtoB':
                kf.get_n_splits(X_A)

            if setup=='AtoA':
                kf.get_n_splits(X_A)

            if setup=='BtoA':
                print('split BtoA')
                kf.get_n_splits(X_B)

            if setup=='AlltoB':
                print('split AlltoB')
                kf.get_n_splits(X_B)

            if setup=='AlltoA':
                print('split AlltoA')
                kf.get_n_splits(X_A)
                
                
                
            for train_index, test_index in tqdm.tqdm(kf.split(X_B)):
                Xtr, Xte = X_B.iloc[train_index], X_A
                Ytr, Yte = Y_B[train_index], Y_A

                reg.fit(Xtr, Ytr)

                pred = reg.predict(Xte)

                PRED = np.array(pred).ravel()
                SCOREMSE.append(mse(Yte, PRED))
                SCOREMAPE.append(mape(Yte, PRED))
                SCORER2.append(r2(Yte,PRED))
                SCORER2ABS.append(r2abs(Yte,PRED))
   )
                SCORECSMAPE.append(csmape(Yte,PRED))
    
    if setup=='BtoB':
        print('folding BtoB')
        for A in range(ATTEMPTS):
            
            
            kf = KFold(n_splits=FOLDS,shuffle=True)
    
            if setup=='AlltoAll':
                kf.get_n_splits(X)

            if setup=='AtoB':
                kf.get_n_splits(X_A)

            if setup=='AtoA':
                kf.get_n_splits(X_A)

            if setup=='BtoA':
                print('split BtoA')
                kf.get_n_splits(X_B)

                
            if setup=='BtoB':
                print('split BtoB')
                kf.get_n_splits(X_B)
                
            if setup=='AlltoB':
                print('split AlltoB')
                kf.get_n_splits(X_B)

            if setup=='AlltoA':
                print('split AlltoA')
                kf.get_n_splits(X_A)
                
                
                
            for train_index, test_index in tqdm.tqdm(kf.split(X_B)):
                Xtr, Xte = X_B.iloc[train_index], X_B.iloc[test_index]
                Ytr, Yte = Y_B[train_index], Y_B[test_index]

                reg.fit(Xtr, Ytr)

                pred = reg.predict(Xte)

                PRED = np.array(pred).ravel()
                SCOREMSE.append(mse(Yte, PRED))
                SCOREMAPE.append(mape(Yte, PRED))
                SCORER2.append(r2(Yte,PRED))
                SCORER2ABS.append(r2abs(Yte,PRED))
                SCORESMAPE.append(smape(Yte,PRED))
                SCORECSMAPE.append(csmape(Yte,PRED))
    
    if setup=='AlltoAll':
        for A in range(ATTEMPTS):
            
            kf = KFold(n_splits=FOLDS,shuffle=True)
    
            if setup=='AlltoAll':
                kf.get_n_splits(X)

            if setup=='AtoB':
                kf.get_n_splits(X_A)

            if setup=='AtoA':
                kf.get_n_splits(X_A)

            if setup=='BtoA':
                print('split BtoA')
                kf.get_n_splits(X_B)
            
            if setup=='BtoB':
                print('split BtoB')
                kf.get_n_splits(X_B)

            if setup=='AlltoB':
                print('split AlltoB')
                kf.get_n_splits(X_B)

            if setup=='AlltoA':
                print('split AlltoA')
                kf.get_n_splits(X_A)
                
                
                
            for train_index, test_index in tqdm.tqdm(kf.split(X)):
                Xtr, Xte = X.iloc[train_index], X.iloc[test_index]
                Ytr, Yte = Y[train_index], Y[test_index]
                
#                 Ytr=np.log1p(Ytr) ###
                
                reg.fit(Xtr, Ytr)

                pred = reg.predict(Xte)
                
#                 pred=np.nan_to_num(np.expm1(pred),0) ###

                PRED = np.array(pred).ravel()
                SCOREMSE.append(mse(Yte, PRED))
                SCOREMAPE.append(mape(Yte, PRED))
                SCORER2.append(r2(Yte,PRED))
                SCORER2ABS.append(r2abs(Yte,PRED))
                SCORESMAPE.append(smape(Yte,PRED))
                SCORECSMAPE.append(csmape(Yte,PRED))

#     RESULTS.append({'Optimizer': REG['name'],
#       'Training': MET['name'],
#       'mape': np.array(SCOREMAPE).mean(),
#       'smape': np.array(SCORESMAPE).mean(), 'csmape': np.array(SCORECSMAPE).mean(),
#       'r2': np.array(SCORER2).mean(), 'r2abs': np.array(SCORER2ABS).mean(),
#       'mse': np.array(SCOREMSE).mean(),'folding':'KF'})
    
    for zz in range(len(SCOREMAPE)):
        RESULTS.append({'Optimizer': REG['name'],
          'Training': MET['name'],
          'mape': SCOREMAPE[zz],
          'smape': SCORESMAPE[zz], 'csmape': SCORECSMAPE[zz],
          'r2': SCORER2[zz], 'r2abs': SCORER2ABS[zz],
          'mse': SCOREMSE[zz],'folding':'KF'})


# In[34]:


RESULTS = []
import tqdm
for O in tqdm.tqdm(OPTIMIZER):
    print(O['name'])
    for M in METRIC:
        print(M['name'])
        print('lpo')
        EVAL2(O,M,setup='AlltoAll')
#         EVAL_LOOCV(O,M)
        print('kfold')
        


# In[ ]:


X,Y = make_regression(n_samples=200, n_features=20, n_informative=5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





------WebKitFormBoundaryWBfL2zhvXYGFDikp--
