# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 22:41:14 2020

@author: sergi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def query_qtd_nulos(coluna):
    return '''
        SELECT {0}, count(*)
        FROM base_original
        where {0} is null
        GROUP BY {0}
        '''.format(coluna)


def query_qtd_categorias(coluna):
    return '''
        SELECT Count(*)
        FROM
        (SELECT distinct {0} coluna
        FROM base_original) distintos
        '''.format(coluna)


URI = 'postgresql+psycopg2://postgres:admin@localhost:5432/seer'

# %%
query = '''
SELECT column_name
  FROM information_schema.columns
 WHERE table_schema = 'public'
   AND table_name   = 'base_original'
     ;
'''

df_colunas = pd.read_sql(query, URI)


# %%
lista_nulos = []
TOTAL_LINHAS = 10450709
for i in range(len(df_colunas)):
    coluna = df_colunas['column_name'][i]
    df_nulos = pd.read_sql(query_qtd_nulos(coluna), URI)
    df_categorias = pd.read_sql(query_qtd_categorias(coluna), URI)
    if df_nulos.shape[0] > 0:
        perc_nulos = df_nulos['count'][0] / TOTAL_LINHAS
        possui_nulo = 'sim'
    else:
        possui_nulo = 'não'
        perc_nulos = 0
    qtd_categorias = df_categorias['count'][0]
    lista_nulos.append([coluna, possui_nulo, perc_nulos, qtd_categorias])
lista_nulos = pd.DataFrame(
    lista_nulos,
    columns=['coluna', 'possui_nulo', 'perc_nulos', 'qtd_categorias'])

# %%
import pickle

with open(r'D:\repositorios\remissia\dicionarios\listas\DASRCM.pkl', 'rb') as arq:
    lista = pickle.load(arq)