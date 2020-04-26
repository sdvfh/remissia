#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
import os
import pandas as pd
import glob
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)
import numpy as np
import cudf
import time


# In[2]:


def merge_rows(x):
    first = x.first_valid_index()
    if first is None:
        return None
    else:
        return x[first]


# In[3]:


cluster = LocalCluster(n_workers=4, threads_per_worker=2, dashboard_address=':8787')
cluster
client = Client(cluster)
# client.run(cudf.set_allocator, "managed", pool=True)


# In[4]:


def tratar_2(db):
    colunas = [
        'EOD10_SZ',
        'CSTUMSIZ',
        'TUMSIZS']
    db['eod_sz'] = db[colunas].apply(merge_rows, axis=1, meta=('eod_sz', 'object'))
    db = db.drop(colunas, axis=1)

    colunas = [
        'EOD10_EX',
        'CSEXTEN']
    db['eod_ex'] = db[colunas].apply(merge_rows, axis=1, meta=('eod_ex', 'object'))
    db = db.drop(colunas, axis=1)

    colunas = [
        'DAJCC7T',
        'DAJCCT',
        'ADJTM_6VALUE',
        'T_VALUE',
        'DASRCT']
    db['value_t'] = db[colunas].apply(merge_rows, axis=1, meta=('value_t', 'object'))
    db = db.drop(colunas, axis=1)

    colunas = [
        'DAJCC7N',
        'DAJCCN',
        'ADJNM_6VALUE',
        'N_VALUE',
        'DASRCN',
    ]
    db['value_n'] = db[colunas].apply(merge_rows, axis=1, meta=('value_n', 'object'))
    db = db.drop(colunas, axis=1)

    colunas = [
        'DAJCC7M',
        'DAJCCM',
        'ADJM_6VALUE',
        'M_VALUE',
        'DASRCM',
    ]
    db['value_m'] = db[colunas].apply(merge_rows, axis=1, meta=('value_m', 'object'))
    db = db.drop(colunas, axis=1)

    colunas = [
        'DAJCC7STG',
        'DAJCCSTG',
        'ADJAJCCSTG',
        'DSRPSG',
        'AJ_3SEER'
    ]
    db['value_sg'] = db[colunas].apply(merge_rows, axis=1, meta=('value_sg', 'object'))
    db = db.drop(colunas, axis=1)

    colunas = ['HST_STGA',
               'SSS77VZ',
               'SSSM2KPZ',
               'DSS1977S',
               'SCSSM2KO',
               'SUMM2K']
    db['summ_stage'] = db[colunas].apply(merge_rows, axis=1, meta=('stage', 'object'))
    db = db.drop(colunas, axis=1)
    return db


# In[5]:


base_paths = '/home/svf/base_col_trat1_quimo'
base_tratada = '/home/svf/base_col_trat2_quimo'


# In[ ]:


for i, base_path in enumerate(glob.iglob(base_paths + '/**/*.csv', recursive=True)):
    antigo = time.time()
    db_sep_path = os.path.join(base_tratada, base_path.split(sep='/')[-2], base_path.split(sep='/')[-1][:-4] + '.csv')
    if os.path.exists(db_sep_path):
        continue
    if not os.path.exists(os.path.dirname(db_sep_path)):
        os.makedirs(os.path.dirname(db_sep_path))
    db = dd.read_csv(base_path, assume_missing=True)
    db = tratar_2(db)
    db.to_csv(db_sep_path, index=False, single_file=True)
    print('Tempo: {0} s | Salvo em: {1}'.format(round(time.time() - antigo, 2), db_sep_path))
    del db

