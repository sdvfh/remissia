import os
import gc
import time
import pickle
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from bayes_opt.util import load_logs
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from category_encoders.cat_boost import CatBoostEncoder
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)


def ler_base():
    logger.info('Lendo base {0}'.format(tipo))
    db = pd.read_csv(
        '../bases/base_col_trat2_quimo_unif/{0}.csv'.format(tipo))
    return db


def retirar_nulos(db, perc_nulo_max):
    logger.info('Retirando colunas com nulos acima de {0} %'.format(
            str(round(perc_nulo_max * 100, 2))))
    colunas_apagar = []
    total_linhas = db.shape[0]
    for coluna in db:
        perc_nulo = db[coluna].isnull().sum() / total_linhas
        if perc_nulo > perc_nulo_max:
            colunas_apagar.append(coluna)
            if debug:
                logger.info('{0:<20} {1} %'.format(
                    coluna, round(perc_nulo * 100, 2)))
                print('{0:<20} {1} %'.format(
                    coluna, round(perc_nulo * 100, 2)))
    db = db.drop(columns=colunas_apagar)
    logger.info('Nulos das linhas retiradas')
    db = db.dropna()
    return


def verif_db_limite(db):
    return db.shape[0] <= 1e4 or db.shape[0] > 1.5e6


def verif_tnm_class(db, tnm_class):
    if round(tnm_class):
        for coluna in ['value_t', 'value_n', 'value_m', 'value_sg']:
            if coluna in db.columns:
                logger.info('Selecionando apenas com classificacao TNM')
                db = db.loc[db[coluna] != 888, :]
                break
    return


def apply_catboost(db, encoder_sigma, encoder_a):
    antigo = time.time()
    encoder = CatBoostEncoder(
        cols=list(db.drop(columns=label).columns),
        drop_invariant=True,
        return_df=True,
        handle_unknown='value',
        handle_missing='value',
        random_state=1,
        sigma=encoder_sigma,
        a=encoder_a)
    db = encoder.fit_transform(db, db[label])
    novo = time.time() - antigo
    logger.info('Utilizando encoder CatBooster: {0} s'.format(
        round(novo, 2)))
    return


def separar_base(db):
    X = db.drop(columns=label).values
    y = db[label].values
    enc_scaler = StandardScaler()
    X = enc_scaler.fit_transform(X, y)

    logger.info('Utilizando Oversampling')
    sm = RandomOverSampler(random_state=42)
    X, y = sm.fit_resample(X, y)

    logger.info('Dividindo base em treino e teste')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=1, stratify=y)
    logger.info('Shape da base: {0}'.format(X_train.shape))
    return X_train, X_test, y_train, y_test


def load_dados_xgb(X_train, X_test, y_train, y_test):
    dtreino = xgb.DMatrix(X_train, label=y_train)
    dteste = xgb.DMatrix(X_test, label=y_test)
    return dtreino, dteste


def treino_ml(params, dtrain, dvalid, dteste):
    antigo = time.time()
    ml = xgb.train(
        params, dtrain, num_boost_round=10000000,
        evals=[
            (dtrain, 'train'),
            (dvalid, 'valid')],
        early_stopping_rounds=20,
        verbose_eval=100)
    tempo = time.time() - antigo
    pred_scores = ml.predict(dteste, ntree_limit=ml.best_ntree_limit)
    return tempo, pred_scores


def modelo_otimizacao(
            eta, max_depth, gamma, colsample_bytree,
            reg_alpha, reg_lambda, min_child_weight,
            max_delta_step, subsample, colsample_bylevel,
            colsample_bynode, scale_pos_weight,
            perc_nulo_max, tnm_class, encoder_sigma, encoder_a):

    antigo_global = time.time()
    db = ler_base()

    try:
        db = retirar_nulos(db, perc_nulo_max)
        if verif_db_limite(db):
            return 0
        db = verif_tnm_class(db, tnm_class)
        db = apply_catboost(db, encoder_sigma, encoder_a)
        X_train, X_test, y_train, y_test = separar_base(db)
        dtreino, dteste = load_dados_xgb(X_train, X_test, y_train, y_test)

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
        count = 1
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
        for train, valid in skf.split(X_train, y_train):
            dtrain = dtreino.slice(train)
            dvalid = dtreino.slice(valid)
            tempo, pred_scores = treino_ml(params, dtrain, dvalid, dteste)
            score_atual = roc_auc_score(y_test, pred_scores, average='micro')
            logger.info(
                'Treinamento {0} | AUCROC: {1} % | Tempo: {2} s'.format(
                    count,
                    round(score_atual * 100, 2),
                    round(tempo, 2)))
            count += 1
            score_list.append(score_atual)
        del dtreino, dteste
        gc.collect()
        novo_global = time.time() - antigo_global
        logger.info('Tempo de treinamento total: {0} min'.format(
            round(novo_global / 60, 2)))
        return np.mean(score_list)
    except Exception as e:
        logger.exception(e)


# %%
label = 'SRV_TIME_MON'
debug = True
lista_tipos = ['BREAST', 'COLRECT', 'DIGOTHR', 'FEMGEN', 'LYMYLEUK', 'MALEGEN',
               'OTHER', 'RESPIR', 'URINARY', 'ALL']

for tipo in lista_tipos:
    logging.basicConfig(
        filename='./logs/treino_ml_gpu.log', level=logging.DEBUG,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(
        '###################### {0} #####################'.format(tipo))
    try:
        modelo = BayesianOptimization(
            modelo_otimizacao, {
                'eta': (0.001, 1),
                'gamma': (0.001, 1),
                'max_depth': (3, 15),
                'reg_alpha': (0, 1),
                'reg_lambda': (0, 1),
                'min_child_weight': (1, 30),
                'max_delta_step': (0, 30),
                'subsample': (0.001, 1),
                'colsample_bylevel': (0.001, 1),
                'colsample_bynode': (0.001, 1),
                'colsample_bytree': (0.001, 1),
                'scale_pos_weight': (0.001, 1),
                'perc_nulo_max': (0.1, 0.8),
                'tnm_class': (0, 1),
                'encoder_sigma': (0, 1),
                'encoder_a': (0.1, 8),
                          })
        if os.path.exists('./logs/ml_grupo_{0}.json'.format(tipo)):
            load_logs(modelo, logs=['./logs/ml_grupo_{0}.json'.format(tipo)])

        logg = JSONLogger(path='./logs/ml_grupo_{0}.json'.format(tipo))
        modelo.subscribe(Events.OPTIMIZATION_STEP, logg)

        modelo.maximize(init_points=50, n_iter=200, acq='ei')
        with open('./bay_files/ml_grupo_{0}.pkl', 'wb') as arq:
            pickle.dump(modelo.max, arq)
    except Exception as e:
        logger.exception(e)
