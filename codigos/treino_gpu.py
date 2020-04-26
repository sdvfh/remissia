#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import glob
import time
import cudf
import cuml
import cupy
import pickle
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import metricas_avaliacao as metric
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours, load_logs
from scikitplot.helpers import binary_ks_curve
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold, train_test_split
cudf.set_allocator("managed", pool=True)


# In[2]:


logging.basicConfig(filename='bay_geral.log', level=logging.DEBUG, 
                    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger=logging.getLogger(__name__)


# In[3]:


pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)


# In[4]:


try:
    base_paths = '/home/svf/base_col_trat2_quimo'
    label = 'SRV_TIME_MON'
    logger.info('Carregando base')
    for i, base_path in enumerate(glob.iglob('/home/svf/base_col_trat2_quimo/**/*.csv', recursive=True)):
        if i == 0:
            db_orig = pd.read_csv(base_path)
        else:
            db_orig = db_orig.append(pd.read_csv(base_path))
except Exception as e:
        logger.error(e)


# In[5]:


try:
    logger.info('Processando base inicial')
    # Retirar pessoas vivas mas sem outro contato além do diagnóstico
    db_orig = db_orig.loc[~((db_orig['STAT_REC'] == 1) & (db_orig['SRV_TIME_MON_FLAG'] == 0)), :]

    # Informações da cirurgia
    c_cirurgia = ['SURGPRIF', 'SURGSCOF', 'SURGSITF', 'NUMNODES', 'NO_SURG', 'SS_SURG', 'SURGSCOP', 'SURGSITE']
    db_orig = db_orig.drop(columns=c_cirurgia)

    # Informações repetidas
    c_repetidos = ['HISTO2V', 'BEHO2V']
    db_orig = db_orig.drop(columns=c_repetidos)
    
    # Retirar colunas com informações de óbito
    c_obitos = ['CODPUB', 'CODPUBKM', 'STAT_REC', 'VSRTSADX', 'VSRTSADX', 'ODTHCLASS', ]
    db_orig = db_orig.drop(columns=c_obitos)

    # Retirar colunas com informações após o diagnóstico
    c_pos_diag = ['REPT_SRC', 'SRV_TIME_MON_FLAG']
    db_orig = db_orig.drop(columns=c_pos_diag)

    # Retirar colunas com informações após o diagnóstico (quimioterapia)
    c_pos_diag_quimo = ['RADIATNR', 'RAD_BRNR', 'RAD_SURG', 'CHEMO_RX_REC']
    db_orig = db_orig.drop(columns=c_pos_diag_quimo)

    # Informações apenas da base
    c_excluir = ['TYPE_FU']
    db_orig = db_orig.drop(columns=c_excluir)
    
    for i in range(4):
        db_orig['PRIMSITE_' + str(i)] = db_orig['PRIMSITE'].str.slice(i, i+1)
    db_orig = db_orig.drop(columns='PRIMSITE')

    db_orig.loc[(db_orig[label] >= 0) & (db_orig[label] < 60), label] = 0
    db_orig.loc[(db_orig[label] >= 60), label] = 1
except Exception as e:
        logger.error(e)


# In[6]:


def search_model(
        eta, max_depth, gamma, colsample_bytree,
        reg_alpha, reg_lambda, min_child_weight,
        max_delta_step, subsample, colsample_bylevel,
        colsample_bynode, scale_pos_weight, pat,
        perc_nulo_max, tnm_class, c_categorias, strategy):
    logger.info('Copiando base original para funcao')
    db = db_orig.copy()
    try:
        #perc_nulo_max = 0.2
        logger.info('Retirando colunas com muitos nulos')
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

        logger.info('Realizando encoder das colunas')
        # encoders = []
        for c in db:
            if c == label:
                continue
            else:
                if len(db[c].unique()) > round(c_categorias):
                    logger.info('{0:<20} | KBinsDiscretizer'.format(c))
                    # print('{0:<20} | KBinsDiscretizer'.format(c))
                    if round(strategy):
                        strategy_final = 'kmeans'
                    else:
                        strategy_final = 'quantile'
                    encoder = KBinsDiscretizer(n_bins=3,
                                               encode='onehot',
                                               strategy=strategy_final)
                else:
                    logger.info('{0:<20} | OneHotEncoder'.format(c))
                    # print('{0:<20} | OneHotEncoder'.format(c))
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                encoder.fit(db[c].values.reshape(-1,1))
                # encoders.append([c, encoder])
                resultado = encoder.transform(db[c].values.reshape(-1,1))
                if isinstance(encoder, KBinsDiscretizer):
                        resultado = resultado.A
                for i in range(resultado.shape[1]):
                    db[c + '_' + str(i)] = resultado[:, i]
                db = db.drop(columns=c)
        logger.info('Dividindo base em treino e teste')
        X_train, X_test, y_train, y_test = train_test_split(db.drop(columns=label), db[label],
                                                            test_size=0.33, random_state=1,
                                                            stratify=db[label])
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
        dtreino = xgb.DMatrix(cudf.from_pandas(X_train), label=cudf.from_pandas(y_train))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
        count = 1
        for train, valid in skf.split(X_train, y_train):
            logger.info('----- Treino {0} -----'.format(count))
            count += 1
            y_valid = y_train.values[valid]
            dtrain = dtreino.slice(train)
            dvalid = dtreino.slice(valid)

            ml = xgb.train(
                params, dtrain, num_boost_round=10000000,
                evals=[(dtrain, 'train'),
                       (dvalid, 'valid')],
                early_stopping_rounds=int(pat),
                verbose_eval=100)

            pred_scores = ml.predict(dvalid, ntree_limit=ml.best_ntree_limit)
            # pred = np.around(pred_scores)
            _, _, _, score_atual, _, _ = binary_ks_curve(y_valid, pred_scores)
            logger.info('KS | {0} %'.format(round(score_atual * 100, 2)))
            #print('KS | {0} %'.format(round(score_atual * 100, 2)))
            score_list.append(score_atual)
        return np.mean(score_list)
    except Exception as e:
        logger.error(e)


# In[ ]:


modelo = BayesianOptimization(
    search_model, {
        'eta': (0.001, 1),
        'gamma': (0.001, 1),
        'max_depth': (3, 30), 
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
        'perc_nulo_max': (0.1, 0.4),
        'tnm_class': (0, 1),
        'c_categorias': (3, 10),
        'strategy': (0, 1)
                  })
if os.path.exists('/home/svf/codigos/bay_geral.json'):
    load_logs(modelo, logs=["./bay_geral.json"])
else:
    logg = JSONLogger(path="./bay_geral.json")
    modelo.subscribe(Events.OPTIMIZATION_STEP, logg)

modelo.maximize(init_points=10, n_iter=50, acq='ei')

