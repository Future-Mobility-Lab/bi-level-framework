from multiprocessing import Pool,Process, Lock,freeze_support
import time
import random

def RSunique(list1):
    # intilize a null list
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def RSlists2dicts(lst):
    dicts = []
    for R in lst:
        di={}
        for k,v in R:
            di[k]=v
        dicts.append(di)
    return dicts

def RSgen(d,iters=100):
    DICT=[]
    import uuid
    
    for i in range(iters):
        x=[]

        for k in sorted(d.keys()):
            x.append([k,random.choice(d[k])])
#         x.append(['id',uuid.uuid4().hex]) 
        DICT.append(x)
        
    DICT = RSunique(DICT)
    DICT = RSlists2dicts(DICT)
    
    return DICT

def RSsample(d):
    return RSgen(d)[0]

from sklearn.metrics import mean_squared_error

def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = y_true[np.nonzero(y_true)] # avoid incidents with zero duration
    y_pred = y_pred[np.nonzero(y_true)]
    return abs(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

def smape(A, F):
    return 100/len(A) * np.sum( np.abs(F - A) / ((np.abs(A) + np.abs(F))/2.0) )

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

def rmsle(yt,yp):
#     return np.sqrt(mean_squared_error(np.log1p(yt),np.log1p(yp)))
    return np.sqrt(((np.log1p(yp) - np.log1p(yt)) ** 2).mean())
#https://xgboost.readthedocs.io/en/latest/parameter.html

#
def csmape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = y_true[np.nonzero(y_true)] # avoid incidents with zero duration
    y_pred = y_pred[np.nonzero(y_true)]
    CS = (np.abs(y_true - y_pred) > (y_true/6))
    y_pred[CS] -= y_true[CS]/6
    y_pred[~CS] = 0
    return np.mean(CS*np.abs((y_true - y_pred) / y_true)) * 100

METRIC = []
from scipy.stats import wasserstein_distance as WD

def logWD(yt,yp):
    return WD(np.log1p(yt),np.log1p(yp))
METRIC.append({'name':'RMSE','instance':rmse,'higherbetter':False})
# METRIC.append({'name':'RMSLE','instance':rmsle,'higherbetter':False})
METRIC.append({'name':'MAPE','instance':mape,'higherbetter':False})
METRIC.append({'name':'R2','instance':r2opt,'higherbetter':False})
METRIC.append({'name':'SMAPE','instance':smape,'higherbetter':False})
# METRIC.append({'name':'WD','instance':WD,'higherbetter':False})




RSOPTIONS={}

RSOPTIONS['used_metrics']=[]
for M in METRIC:
    RSOPTIONS['used_metrics'].append(M['name'])

RSOPTIONS['resutls']=None
RSOPTIONS['best']=None
RSOPTIONS['model']=None
RSOPTIONS['X']=None
RSOPTIONS['Y']=None
for M in METRIC:
    RSOPTIONS['best_'+M['name']] = None
    RSOPTIONS['metric_'+M['name']] = M['instance']
    RSOPTIONS[M['name']+'hb'] = M['higherbetter']
    

from sklearn.model_selection import KFold
import numpy as np



def RSproc(elt):
    time.sleep(random.random()/100)
    
    kf = KFold(6,shuffle=True)
    kf.get_n_splits(RSOPTIONS['X'])
    
    
    SCORE={}
    
    for k in RSOPTIONS.keys():
            if k.startswith('metric'):
                SCORE[k] = []
#     print(SCORE)
    for train_index, test_index in kf.split(RSOPTIONS['X']):
        Xtr, Xte = RSOPTIONS['X'].iloc[train_index], RSOPTIONS['X'].iloc[test_index]
        Ytr, Yte = RSOPTIONS['Y'][train_index], RSOPTIONS['Y'][test_index]
        reg = RSOPTIONS['model'](**elt)
#         print(reg)
        reg.fit(Xtr,Ytr)
        
        for k in RSOPTIONS.keys():
#             print(k)
            if k.startswith('metric'):
#                 print('!'+k)
#                 print(k)
                res = RSOPTIONS[k](Yte,reg.predict(Xte))
                
#                 if 'r2' in k:
#                     res = 1-res
                    
                SCORE[k].append(res)
#     print(SCORE)
    return SCORE

RSextra=False
RSintra=False
RSORtype='LOF'

        
def RSprocEX(elt):
	
    print(RSOPTIONS['X'])
	
    time.sleep(random.random()/100)
    
    kf = KFold(5,shuffle=True)
#     kf.get_n_splits(RSOPTIONS['X'])

    import copy
    X,Y = copy.deepcopy(RSOPTIONS['X'].values), copy.deepcopy(RSOPTIONS['Y'])
    

#     print(Y.max())
    if RSextra or RSintra:
        TODELETE=[]
        
        LOFparams={}
        for P in elt.keys():
            if P.startswith('LOF'):
                LOFparams[P.replace('LOF','')] = elt[P]
                TODELETE.append(P)
        
        IFparams={}
        for P in elt.keys():
            if P.startswith('IF'):
                IFparams[P.replace('IF','')] = elt[P]
                TODELETE.append(P)
                
        for P in TODELETE:
            del elt[P]
            
    if RSextra:
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.ensemble import IsolationForest
        clf=None
        
        if RSORtype=='LOF':
#             print(LOFparams)
            clf = LocalOutlierFactor(**LOFparams)
        if RSORtype=='IF':
#             print(IFparams)
            clf = IsolationForest(**IFparams)
            
        anomaly = clf.fit_predict(np.concatenate([X,Y.reshape(-1,1)],axis=1))
        X=X[anomaly>0]
        Y=Y[anomaly>0]
        
    


    
    X_A = X[Y<RSOPTIONS['TH']]
    Y_A = Y[Y<RSOPTIONS['TH']]
    
    X_B = X[Y>=RSOPTIONS['TH']]
    Y_B = Y[Y>=RSOPTIONS['TH']]
    
#     print(len(X),len(X_A),len(X_B),RSOPTIONS['TH'],Y[:5])
    
    transformer=None
    if RStransform=='LOG':
#         from sklearn.preprocessing import QuantileTransformer as QT
#         transformer=QT()
        Y = np.log1p(Y) #transformer.fit_transform(Y.reshape(-1,1)).ravel()
        Y_A = np.log1p(Y_A)
        Y_B = np.log1p(Y_B)
    
    SCORE={}
    
    for k in RSOPTIONS.keys():
            if k.startswith('metric'):
                SCORE[k] = []
#     print(SCORE)


    if RSOPTIONS['scenario']=='AlltoAll':
        kf.get_n_splits(X)

    if RSOPTIONS['scenario']=='AtoB':
        kf.get_n_splits(X_A)

    if RSOPTIONS['scenario']=='AtoA':
        kf.get_n_splits(X_A)

    if RSOPTIONS['scenario']=='BtoA':
        kf.get_n_splits(X_B)

    if RSOPTIONS['scenario']=='AlltoB':
        kf.get_n_splits(X_B)

    if RSOPTIONS['scenario']=='AlltoA':
        kf.get_n_splits(X_A)
        
    ORclf=None
    
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.ensemble import IsolationForest
    
    if RSintra and RSORtype=='LOF':
#                 print(LOFparams)
        ORclf = LocalOutlierFactor(**LOFparams)
    
    if RSintra and RSORtype=='IF':
#                 print(IFparams)
        ORclf = IsolationForest(**IFparams)
    
    def check(Xtr,Ytr,Xte,Yte):
#         print(len(Xtr),len(Xte))
        if RSintra:


            anomaly = ORclf.fit_predict(np.concatenate([Xtr,Ytr.reshape(-1,1)],axis=1))
            Xtr=Xtr[anomaly>0]
            Ytr=Ytr[anomaly>0]
        
        reg = RSOPTIONS['model'](**elt)
        

        reg.fit(Xtr,Ytr)

        for k in RSOPTIONS.keys():
            if k.startswith('metric'):
                
                res=None
                if RStransform=='LOG':
                    res = RSOPTIONS[k](np.nan_to_num(np.expm1(Yte),0), np.nan_to_num(np.expm1(reg.predict(Xte).ravel()),0) )
#                     print(k,res)
                else:  
                    res = RSOPTIONS[k](Yte,reg.predict(Xte).ravel())
                
#                 print(np.expm1(Yte[:3]),np.expm1(reg.predict(Xte))[:3])
                
                SCORE[k].append(res)
    import pandas as pd
    import tqdm
    
    if RSOPTIONS['scenario']=='AlltoA':
        
        for train_index, test_index in kf.split(X_A):
            Xtr, Xte = np.concatenate([X_B,X_A[train_index]], axis=0), X_A[test_index]
            Ytr, Yte = np.concatenate([np.array(Y_B), np.array(Y_A)[train_index]],axis=0), Y_A[test_index]
            check(Xtr,Ytr,Xte,Yte)
            
    if RSOPTIONS['scenario']=='AlltoB':        
        for train_index, test_index in kf.split(X_B):
            Xtr, Xte = np.concatenate([X_A,X_B[train_index]], axis=0), X_B[test_index]
            Ytr, Yte = np.concatenate([np.array(Y_A), np.array(Y_B)[train_index]],axis=0), Y_B[test_index]
            check(Xtr,Ytr,Xte,Yte)
            
    if RSOPTIONS['scenario']=='AtoA': 
        for train_index, test_index in kf.split(X_A):
            Xtr, Xte = X_A[train_index], X_A[test_index]
            Ytr, Yte = Y_A[train_index], Y_A[test_index]                   
            check(Xtr,Ytr,Xte,Yte)
                
    if RSOPTIONS['scenario']=='AtoB':            
        for train_index, test_index in kf.split(X_A):
            Xtr, Xte = X_A[train_index], X_B
            Ytr, Yte = Y_A[train_index], Y_B
            check(Xtr,Ytr,Xte,Yte)
                
    if RSOPTIONS['scenario']=='BtoA':             
        for train_index, test_index in kf.split(X_B):
            Xtr, Xte = X_B[train_index], X_A
            Ytr, Yte = Y_B[train_index], Y_A
            check(Xtr,Ytr,Xte,Yte)
            
    if RSOPTIONS['scenario']=='BtoB':       
        for train_index, test_index in kf.split(X_B):
            Xtr, Xte = X_B[train_index], X_B[test_index]
            Ytr, Yte = Y_B[train_index], Y_B[test_index]
            check(Xtr,Ytr,Xte,Yte)
            
    if RSOPTIONS['scenario']=='AlltoAll':         
        for train_index, test_index in kf.split(X):
            Xtr, Xte = X[train_index], X[test_index]
            Ytr, Yte = Y[train_index], Y[test_index]
            check(Xtr,Ytr,Xte,Yte)    

    return SCORE




global RSarray

RStransform='None'

Isteps=np.array([0.01,0.02,0.03,0.04,0.05])/6 #number of folds
Esteps=np.array([0.01,0.02,0.03,0.04,0.05]) #number of folds

def getsteps(s):
    if s=='intra':
        return Isteps
    if s=='extra':
        return Esteps

def RandomSearchX(X,Y,model=None,params=None,iters=100,jobs=26,TH=45,scenario='AlltoAll',extra=False, intra=False,OR='LOF',transform='None'):
    print('RandomSearch:',model,iters)
    
    global RSOPTIONS,RSextra,RSintra,RSORtype,RStransform
    
    RStransform=transform
    
    RSOPTIONS['X']=X
    RSOPTIONS['Y']=Y
    RSOPTIONS['model']=model
    RSOPTIONS['params'] = params
    RSOPTIONS['TH']=TH
    RSOPTIONS['scenario']=scenario
    
    freeze_support()
    if extra or intra:
        RSextra=extra
        RSintra=intra
        RSORtype=OR
        
        if RSORtype=='LOF':
            params.update({
               'LOFn_neighbors':np.arange(2,31),'LOFmetric':['cityblock', 'euclidean', 'l1', 'l2', 'manhattan'],
               'LOFalgorithm':['auto', 'ball_tree', 'kd_tree', 'brute'], 'LOFcontamination':getsteps('intra') if intra else getsteps('extra')})
        if RSORtype=='IF':
            
            params.update({
                'IFcontamination':getsteps('intra') if intra else getsteps('extra'),
                'IFbehaviour':['new'],
                'IFn_estimators':np.arange(20,205,5)
                })
    
    RSarray = RSgen(params,iters)
    start = time.time()
    RSscores=None
    with Pool(processes=jobs) as pool:
        RSscores=pool.map(RSprocEX, RSarray)
#     print(RSscores)
    
    RSOPTIONS['time'] = np.round(time.time()-start,2)
    BEST={}
    
    for k in RSOPTIONS.keys():
        if k.startswith('metric'):
            
            metric_results = list(map(lambda v: np.mean(v[k]), RSscores))
            
#             print(k,metric_results)
            
            RSresults = list(sorted(zip(metric_results,RSarray),key=lambda key:key[0],reverse=RSOPTIONS[k.split('_')[1]+'hb']))
            BEST['best_'+k] = RSresults[0]
    RSOPTIONS.update(BEST)
    return RSOPTIONS

def RSeval(elt,scenario='AlltoAll'):
    res = RSprocEX(elt)
    nres={}
    for k in res.keys():
        nres[k.split('_')[1]] = np.mean(res[k])
    return nres




#provide final result using multiple metrics
#add also wesserstein distance

#Joint outlier removal, joint ELM, joint NGBoost

#Embed metrics within code and do automatic higherbetter?

#provide results for multiple scenarios??


#any model with fit and predict now can be used












# RSresults=None
# RSbest=None
# RSmodelclass=None
# RSX=None
# RSY=None
# RSlocalmetric=None





# def RSprocOR(elt):
#     time.sleep(random.random()/100)
    
#     kf = KFold(8,shuffle=True) #574/5 = 114
#     kf.get_n_splits(RSOPTIONS['X'])
    
    
#     Aparams={}
    
#     for J in RSarray:
#         if elt['id']==J['id']:
            
#             for P in elt.keys(): 
#                     if P.startswith('AA'):
#                         Aparams[P.replace('AA','')] = elt[P]
#             break
            
#     from sklearn.ensemble import IsolationForest
#     from sklearn.neighbors import LocalOutlierFactor
#     clf = LocalOutlierFactor(**Aparams)
#     ou = clf.fit_predict(RSOPTIONS['X'])
#     locX = RSOPTIONS['X'].values[ou>0]
#     locY = RSOPTIONS['Y'][ou>0]
        
    
#     SCORE=[]
#     for train_index, test_index in kf.split(locX):
#         Xtr, Xte = locX[train_index], locX[test_index]
#         Ytr, Yte = locY[train_index], locY[test_index]
        
#         reg = RSOPTIONS['model'](**elt)
        
#         from sklearn.ensemble import IsolationForest
#         from sklearn.neighbors import LocalOutlierFactor
        
        
        
        
#         if RSOPTIONS['logY']:
#             reg.fit(Xtr,np.log1p(Ytr))
#         else:    
#             reg.fit(Xtr,Ytr)
        
#         if RSOPTIONS['logY']:
#             SCORE.append(RSOPTIONS['metric'](Yte, np.nan_to_num(np.expm1(reg.predict(Xte)),-1) ))
#         else:
#             SCORE.append(RSOPTIONS['metric'](Yte,reg.predict(Xte)))
        
#     return sum(SCORE)/8

# global RSarray
# def RandomSearch(X,Y,model=None,params=None,metric=None,higherbetter=False,iters=100,jobs=26,logY=False,hardpass=None):
#     print('RandomSearch:',model,iters)
    
#     global RSOPTIONS
    
#     RSOPTIONS['metric']=metric
    
#     RSOPTIONS['X']=X
#     RSOPTIONS['Y']=Y
#     RSOPTIONS['model']=model
#     RSOPTIONS['logY']=logY
#     RSOPTIONS['params'] = params
    
#     freeze_support()
    
#     #if not hardpass.keys() in params.keys(): error
    
#     if hardpass:
#         for k in hardpass.keys():
#             params[k] = hardpass[k]
    
#     RSarray = RSgen(params,iters)
#     start = time.time()
    
#     with Pool(processes=jobs) as pool:
#         RSscores=pool.map(RSproc, RSarray)
        
#     RSOPTIONS['time'] = np.round(time.time()-start,2)

#     RSresults=None

#     if higherbetter:
#         RSresults = list(sorted(zip(RSscores,RSarray),key=lambda key:key[0],reverse=True))
#     else:
#         RSresults = list(sorted(zip(RSscores,RSarray),key=lambda key:key[0],reverse=False))

# #     RSbest = RSresults[0]
#     RSOPTIONS['results'] = RSresults
#     RSOPTIONS['best'] = RSresults[0]
    
#     return RSOPTIONS['best']




# def RandomSearchOR(X,Y,model=None,params=None,metric=None,higherbetter=False,iters=100,jobs=26,logY=False,hardpass=None):
#     print('RandomSearch:',model,iters)
#     global RSarray
#     global RSOPTIONS
    
#     RSOPTIONS['metric']=metric
    
#     RSOPTIONS['X']=X
#     RSOPTIONS['Y']=Y
#     RSOPTIONS['model']=model
#     RSOPTIONS['logY']=logY
#     RSOPTIONS['params'] = params
    
#     freeze_support()
    
#     #if not hardpass.keys() in params.keys(): error
    
#     if hardpass:
#         for k in hardpass.keys():
#             RSOPTIONS['params'][k] = hardpass[k]
    
#     if hardpass:
#         for k in hardpass.keys():
#             params[k] = hardpass[k]
    
#     RSarray = RSgen(params,iters)
#     start = time.time()
#     with Pool(processes=jobs) as pool:
#         RSscores=pool.map(RSprocOR, RSarray)
#     RSOPTIONS['time'] = np.round(time.time()-start,2)

#     RSresults=None

#     if higherbetter:
#         RSresults = list(sorted(zip(RSscores,RSarray),key=lambda key:key[0],reverse=True))
#     else:
#         RSresults = list(sorted(zip(RSscores,RSarray),key=lambda key:key[0],reverse=False))

# #     RSbest = RSresults[0]
#     RSOPTIONS['results'] = RSresults
#     RSOPTIONS['best'] = RSresults[0]
    
#     return RSOPTIONS['best']
