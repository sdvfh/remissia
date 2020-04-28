import os
import glob
import time
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)


# %%
def merge_rows(x):
    '''Funcao para juntar as colunas por linha.
    A ordem das colunas no dataframe que rege a preferencia.'''
    first = x.first_valid_index()
    if first is None:
        return None
    else:
        return x[first]


# %%
cluster = LocalCluster(n_workers=4, threads_per_worker=2,
                       dashboard_address=':8787')
client = Client(cluster)
# client.run(cudf.set_allocator, "managed", pool=True)


# %%
def tratar_principal(db):
    '''Parte 3 do pre-processamento da base.
    Uniao das colunas com as mesmas informacoes.'''
    colunas_final = ['eod_sz', 'eod_ex', 'value_t', 'value_n', 'value_m',
                     'value_sg', 'summ_stage']
    colunas_merge = [
        ['EOD10_SZ', 'CSTUMSIZ', 'TUMSIZS'],
        ['EOD10_EX', 'CSEXTEN'],
        ['DAJCC7T', 'DAJCCT', 'ADJTM_6VALUE', 'T_VALUE', 'DASRCT'],
        ['DAJCC7N', 'DAJCCN', 'ADJNM_6VALUE', 'N_VALUE', 'DASRCN'],
        ['DAJCC7M', 'DAJCCM', 'ADJM_6VALUE', 'M_VALUE', 'DASRCM'],
        ['DAJCC7STG', 'DAJCCSTG', 'ADJAJCCSTG', 'DSRPSG', 'AJ_3SEER'],
        ['HST_STGA', 'SSS77VZ', 'SSSM2KPZ', 'DSS1977S', 'SCSSM2KO', 'SUMM2K']]
    colunas_zip = list(zip(colunas_final, colunas_merge))

    for coluna_final, colunas_merge in colunas_zip:
        db[coluna_final] = db[colunas_merge].apply(
            merge_rows, axis=1, meta=(coluna_final, 'object'))
        db = db.drop(colunas_merge, axis=1)
    return db


# %%
dir_base_origem = 'base_col_trat1_quimo'
dir_base_destino = 'base_col_trat2_quimo'


# %%
for i, path_base in enumerate(glob.iglob(
        '../bases/{0}/**/*.csv'.format(dir_base_origem),
        recursive=True)):
    antigo = time.time()
    path_base_tratada = os.path.join(
        '../bases/{0}'.format(dir_base_destino),
        path_base.split(sep='/')[-2],
        path_base.split(sep='/')[-1][:-4] + '.csv')

    if os.path.exists(path_base_tratada):
        continue
    if not os.path.exists(os.path.dirname(path_base_tratada)):
        os.makedirs(os.path.dirname(path_base_tratada))

    db = dd.read_csv(path_base, assume_missing=True)
    db = tratar_principal(db)
    db.to_csv(path_base_tratada, index=False, single_file=True)
    print('Tempo: {0} s | Salvo em: {1}'.format(
        round(time.time() - antigo, 2), path_base_tratada))
    del db
