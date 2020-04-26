#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gc
import sys
import glob
import time
import cudf
import cuml
import cupy
import GPUtil
import pickle
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import metricas_avaliacao as metric
from bayes_opt.event import Events
from multiprocessing import Process
from bayes_opt.logger import JSONLogger
from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours, load_logs
from scikitplot.helpers import binary_ks_curve
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
# cudf.set_allocator("managed")


# In[2]:


pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)


# In[3]:


label = 'SRV_TIME_MON'


# In[4]:


gpu = GPUtil.getGPUs()[0]


# In[5]:


def search_model(
            eta, max_depth, gamma, colsample_bytree,
            reg_alpha, reg_lambda, min_child_weight,
            max_delta_step, subsample, colsample_bylevel,
            colsample_bynode, scale_pos_weight, pat,
            perc_nulo_max, tnm_class, c_categorias, strategy):
    logger.info('Lendo base {0}'.format(tipo))
    db = pd.read_csv('/home/svf/base_col_trat2_quimo_unif/{0}.csv'.format(tipo))
    try:
        #perc_nulo_max = 0.2
        logger.info('Retirando colunas com nulos acima de {0} %'.format(str(round(perc_nulo_max * 100, 2))))
        c_apagar = []
        total_linhas = db.shape[0]
        for c in db:
            perc_nulo = db[c].isnull().sum() / total_linhas
            if perc_nulo > perc_nulo_max:
                logger.info('{0:<20} {1} %'.format(c, round(perc_nulo * 100, 2)))
                #print('{0:<20} {1} %'.format(c, round(perc_nulo * 100, 2)))
                c_apagar.append(c)
        db = db.drop(columns=c_apagar)
        logger.info('Retirando os nulos das linhas')
        db = db.dropna()

        if round(tnm_class):
            for c in ['value_t', 'value_n', 'value_m', 'value_sg']:
                if c in db.columns:
                    logger.info('Selecionando apenas com classificacao TNM')
                    db = db.loc[db[c] != 888, :]
                    break

        
        logger.info('Realizando encoder das colunas (OneHot acima de {0} categorias)'.format(str(int(c_categorias))))
        # encoders = []
        encoder = OneHotEncoder(sparse=True, handle_unknown='ignore')
encoder.fit(db_temp[c].values.reshape(-1,1))
# encoders.append([c, encoder])
resultado = encoder.transform(db_temp[c].values.reshape(-1,1))
        logger.info('Utilizando Oversampling')
        sm = RandomOverSampler(random_state=42)
        X, y = sm.fit_resample(db.drop(columns=label).values, db[label].values)
        logger.info('Dividindo base em treino e teste')
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.33, random_state=1,
                                                            stratify=y)

        params = dict(
                max_depth=int(max_depth),
                learning_rate=eta,
                verbosity=0, 
                objective='binary:logistic', 
                booster='gbtree',
                n_jobs=-1, 
                gamma=gamma, 
                min_child_weight=int(min_child_weight), 
                max_delta_step=int(max_delta_step), 
                subsample=subsample, 
                colsample_bytree=colsample_bytree,
                colsample_bylevel=colsample_bylevel,
                colsample_bynode=colsample_bynode,
                reg_alpha=reg_alpha, 
                reg_lambda=reg_lambda,
                scale_pos_weight=scale_pos_weight, 
                random_state=0,
                tree_method='gpu_hist',
                eval_metric='rmsle')

        score_list = []
        # logger.info('Inserindo base na memoria da GPU')
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
        count = 1
        dtreino = xgb.DMatrix(X_train, label=y_train)
        for train, valid in skf.split(X_train, y_train):
            dtrain = dtreino.slice(train)
            dvalid = dtreino.slice(valid)
            logger.info('Shape da base: {0}'.format(X_train.shape))
            logger.info('Treinamento {0} | Inicializado'.format(count))
            logger.info('Memoria GPU: {0} %'.format(round(gpu.memoryUtil * 100, 2)))
            ml = xgb.train(
                params, dtrain, num_boost_round=10000000,
                evals=[(dtrain, 'train'),
                       (dvalid, 'valid')],
                early_stopping_rounds=int(pat),
                verbose_eval=100)
            logger.info('Treinamento {0} | Finalizado'.format(count))
            logger.info('Memoria GPU antes de apagar: {0} %'.format(round(gpu.memoryUtil * 100, 2)))
            pred_scores = ml.predict(dvalid, ntree_limit=ml.best_ntree_limit)
            del ml, dtrain, dvalid
            logger.info('Memoria GPU depois de apagar: {0} %'.format(round(gpu.memoryUtil * 100, 2)))
            y_valid = y_train[valid]
            # pred = np.around(pred_scores)
            score_atual = roc_auc_score(y_valid, pred_scores, average='micro')
            # _, _, _, score_atual, _, _ = binary_ks_curve(y_valid, pred_scores)
            logger.info('Treinamento {0} | AUCROC: {1} %'.format(count, round(score_atual * 100, 2)))
            count += 1
            #print('KS | {0} %'.format(round(score_atual * 100, 2)))
            score_list.append(score_atual)
        dtreino.__del__()
        gc.collect()
        return np.mean(score_list)
    except Exception as e:
        logger.error(e)


# In[ ]:


for tipo in ['BREAST', 'COLRECT', 'DIGOTHR', 'FEMGEN', 'LYMYLEUK', 'MALEGEN', 'OTHER', 'RESPIR', 'URINARY']:
    logging.basicConfig(filename='bay_esp.log', level=logging.DEBUG, 
                    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    logger=logging.getLogger(__name__)
    logger.info('########### {0} ##########'.format(tipo))
    try:
        modelo = BayesianOptimization(
            search_model, {
                'eta': (0.001, 1),
                'gamma': (0.001, 1),
                'max_depth': (3, 40), 
                'reg_alpha': (0, 1),
                'reg_lambda': (0, 1),
                'min_child_weight': (1, 30),
                'max_delta_step': (0, 30),
                'subsample': (0.001, 1),
                'colsample_bylevel': (0.001 ,1),
                'colsample_bynode': (0.001, 1),
                'colsample_bytree': (0.001, 1),
                'scale_pos_weight': (0.001, 1),
                'pat': (3, 25),
                'perc_nulo_max': (0.1, 0.7),
                'tnm_class': (0, 1),
                'c_categorias': (5, 20),
                'strategy': (0, 1)
                          })
        if os.path.exists('/home/svf/codigos/bay_esp_{0}.json'.format(tipo)):
            load_logs(modelo, logs=["./bay_esp_{0}.json".format(tipo)])

        logg = JSONLogger(path="./bay_esp_{0}.json".format(tipo))
        modelo.subscribe(Events.OPTIMIZATION_STEP, logg)

        modelo.maximize(init_points=50, n_iter=200, acq='ei')
        with open('/home/svf/codigos/bay_files/bay_file_{0}.pkl', 'wb') as arq:
            pickle.dump(modelo.max, arq)
    except Exception as e:
        logger.error(e)
        


# In[ ]:




