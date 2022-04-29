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

from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score



def f1_macro(yt,yp):
	return f1_score(yt,yp,average='macro')

METRIC = []

METRIC.append({'name':'F1','instance':f1_macro,'higherbetter':True})
# METRIC.append({'name':'RMSLE','instance':rmsle,'higherbetter':False})
#~ METRIC.append({'name':'MAPE','instance':mape,'higherbetter':False})
#~ METRIC.append({'name':'R2','instance':r2opt,'higherbetter':False})
#~ METRIC.append({'name':'SMAPE','instance':smape,'higherbetter':False})
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

    for train_index, test_index in kf.split(RSOPTIONS['X']):
        Xtr, Xte = RSOPTIONS['X'].iloc[train_index], RSOPTIONS['X'].iloc[test_index]
        Ytr, Yte = RSOPTIONS['Y'][train_index], RSOPTIONS['Y'][test_index]
        reg = RSOPTIONS['model'](**elt)

        reg.fit(Xtr,Ytr)
        
        for k in RSOPTIONS.keys():

            if k.startswith('metric'):

                res = RSOPTIONS[k](Yte,reg.predict(Xte))

                SCORE[k].append(res)
    return SCORE

RSextra=False
RSintra=False
RSORtype='LOF'

     

global RSarray

RStransform='None'

Isteps=np.array([0.01,0.02,0.03,0.04,0.05])/6 #number of folds
Esteps=np.array([0.01,0.02,0.03,0.04,0.05]) #number of folds

def getsteps(s):
    if s=='intra':
        return Isteps
    if s=='extra':
        return Esteps

def RandomSearchC(X,Y,model=None,params=None,iters=100,jobs=26,TH=45,scenario='AlltoAll',extra=False, intra=False,OR='LOF',transform='None'):
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
        RSscores=pool.map(RSproc, RSarray)
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
    res = RSproc(elt)
    nres={}
    for k in res.keys():
        nres[k.split('_')[1]] = np.mean(res[k])
    return nres
