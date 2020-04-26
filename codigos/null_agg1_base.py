#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import glob
import pandas as pd
import numpy as np
import cudf
cudf.set_allocator('managed', pool=True)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)


# In[2]:


path_home = '/home/svf'
dir_base_origem = 'base_col_sep_quimo'
dir_base_destino = 'base_col_trat1_quimo'


# In[3]:


path_base_destino = os.path.join(path_home, dir_base_origem)
path_dic_variaveis = os.path.join(home_path, 'config', 'variaveis_quimo.csv')
dic_variaveis = pd.read_csv(path_dic_variaveis, sep=';')


# In[4]:


def tratar_1(db):
    dic_dados = [
    ['pX', ''],
    ['p0', '0'],
    ['pA', '10'],
    ['pISU', '60'],
    ['pISD', '70'],
    ['pIS', '50'],
    ['p1MI', '110'],
    ['p1A1', '130'],
    ['p1A2', '140'],
    ['p1A', '120'],
    ['p1B1', '160'],
    ['p1B2', '170'],
    ['p1B', '150'],
    ['p1C', '180'],
    ['p1D', '181'],
    ['p1', '100'],
    ['p2A1', '211'],
    ['p2A2', '212'],
    ['p2A', '210'],
    ['p2B', '220'],
    ['p2C', '230'],
    ['p2D', '240'],
    ['p2', '200'],
    ['p3A', '310'],
    ['p3B', '320'],
    ['p3C', '330'],
    ['p3D', '340'],
    ['p3', '300'],
    ['p4A', '410'],
    ['p4B', '420'],
    ['p4C', '430'],
    ['p4D', '440'],
    ['p4E', '450'],
    ['p4', '400'],
    ['88', '888'],
    ['cX', ''],
    ['c0', '0'],
    ['cA', '10'],
    ['cISU', '60'],
    ['cISD', '70'],
    ['cIS', '50'],
    ['c1MI', '110'],
    ['c1A1', '130'],
    ['c1A2', '140'],
    ['c1A', '120'],
    ['c1B1', '160'],
    ['c1B2', '170'],
    ['c1B', '150'],
    ['c1C', '180'],
    ['c1', '100'],
    ['c1D', '181'],
    ['c2A1', '211'],
    ['c2A2', '212'],
    ['c2A', '210'],
    ['c2B', '220'],
    ['c2C', '230'],
    ['c2D', '240'],
    ['c2', '200'],
    ['c3A', '310'],
    ['c3B', '320'],
    ['c3C', '330'],
    ['c3D', '340'],
    ['c3', '300'],
    ['c4A', '410'],
    ['c4B', '420'],
    ['c4C', '430'],
    ['c4D', '440'],
    ['c4E', '450'],
    ['c4', '400'],
    ]

    for orig, mod in dic_dados:
        db['DASRCT'] = db['DASRCT'].str.replace(orig, mod)

    dic_dados = [
        ['cX', ''],
        ['c0A', '1'],
        ['c0B', '2'],
        ['c0', '0'],
        ['c1A', '110'],
        ['c1B', '120'],
        ['c1C', '130'],
        ['c1', '100'],
        ['c2A', '210'],
        ['c2B', '220'],
        ['c2C', '230'],
        ['c2', '200'],
        ['c3A', '310'],
        ['c3B', '320'],
        ['c3C', '330'],
        ['c3', '300'],
        ['c4', '400'],
        ['88', '888'],
        ['pX', ''],
        ['p0I-', '10'],
        ['p0I+', '20'],
        ['p0M-', '30'],
        ['p0M+', '40'],
        ['p0A', '1'],
        ['p0B', '2'],
        ['p0', '0'],
        ['p1A', '110'],
        ['p1B', '120'],
        ['p1C', '130'],
        ['p1MI', '180'],
        ['p1', '100'],
        ['p2A', '210'],
        ['p2B', '220'],
        ['p2C', '230'],
        ['p2', '200'],
        ['p3A', '310'],
        ['p3B', '320'],
        ['p3C', '330'],
        ['p3', '300'],
        ['p4', '400'],
    ]

    for orig, mod in dic_dados:
        db['DASRCN'] = db['DASRCN'].str.replace(orig, mod)
    db.loc[db['DASRCN'] == '10-', 'DASRCN'] = '10'
    db.loc[db['DASRCN'] == '10+', 'DASRCN'] = '10'
    db.loc[db['DASRCN'] == '20+', 'DASRCN'] = '20'
    db.loc[db['DASRCN'] == '30-', 'DASRCN'] = '30'
    db.loc[db['DASRCN'] == '30+', 'DASRCN'] = '30'
    db.loc[db['DASRCN'] == '40+', 'DASRCN'] = '40'

    dic_dados = [
        ['c0I+', '10'],
        ['c0', '0'],
        ['c1A', '110'],
        ['c1B', '120'],
        ['c1C', '130'],
        ['c1D', '140'],
        ['c1E', '150'],
        ['c1', '100'],
        ['p1A', '110'],
        ['p1B', '120'],
        ['p1C', '130'],
        ['p1D', '140'],
        ['p1E', '150'],
        ['p1', '100'],
        ['88', '888'],
    ]

    for orig, mod in dic_dados:
        db['DASRCM'] = db['DASRCM'].str.replace(orig, mod)
    db.loc[db['DASRCM'] == '10+', 'DASRCM'] = '10'

    dic_dados = [
        ['0A', '10'],
        ['0IS', '20'],
        ['1A1', '130'],
        ['1A2', '140'],
        ['1A', '120'],
        ['1B1', '160'],
        ['1B2', '170'],
        ['1B', '150'],
        ['1C', '180'],
        ['1S', '190'],
        ['2A1', '322'],
        ['2A2', '324'],
        ['2A', '320'],
        ['2B', '330'],
        ['2C', '340'],
        ['3A', '520'],
        ['3B', '530'],
        ['3C1', '541'],
        ['3C2', '542'],
        ['3C', '540'],
        ['4A1', '722'],
        ['4A2', '724'],
        ['4A', '720'],
        ['4B', '730'],
        ['4C', '740'],
        ['OC', '900'],
        ['88', '888'],
        ['99', ''],
    ]

    for orig, mod in dic_dados:
        db['DSRPSG'] = db['DSRPSG'].str.replace(orig, mod)

    db.loc[db['DSRPSG'].str.strip() == '', 'DSRPSG'] = None

    db['DSRPSG'] = db['DSRPSG'].astype(float)

    values = [   
        [4, 700],
        [3, 500],
        [1, 100],
        [2, 300],

        ]
    for orig, mod in values:
        db.loc[db['DSRPSG'] == orig, 'DSRPSG'] = mod

    v_continuo = []
    for linha in dic_variaveis.values:
        nome = linha[0].strip()
        c_apagar = linha[1]
        c_continuo = linha[2]
        num_null = linha[3]

        if nome == 'DSRPSG':
            continue

        if c_apagar == 'x':
            db = db.drop(columns=nome)
            continue

        elif c_continuo == 'x':
            v_continuo.append(nome)

        if isinstance(num_null, str):
            if num_null.isdigit():
                db[nome] = db[nome].str.replace(num_null, '')
            elif len(num_null.split(',')) > 1:
                for num_null_temp in num_null.split(','):
                    db[nome] = db[nome].str.replace(num_null_temp, '')
            else:
                db[nome] = db[nome].str.replace(num_null, '')

        db.loc[db[nome].str.strip() == '', nome] = None

        if all(db[nome][db[nome].notnull()].str.isdigit()) or (db[nome][db[nome] != ''].shape[0] == 0):
                db[nome] = db[nome].astype(float)

    db = db.drop(columns=
                 ['eod13_sz',
                  'eod13_psv',
                  'eod13_ex',
                  'eod13_ss',
                  'eod13_nd',
                  'eod13_dnd',
                  'eod13_dss',
                  'EOD2',
                  'eod4_sz',
                  'eod4_extent',
                  'edo4_nd',])

    def add_digito(num):
        if num % 10 == 9:
            return num * 10 + 9
        elif num == 88:
            return num * 10 + 8
        else:
            return num * 10

    colunas = [
        'DAJCCT',
        'ADJTM_6VALUE',
        'T_VALUE',
        'DAJCCN',
        'ADJNM_6VALUE',
        'N_VALUE',
        'DAJCCM',
        'ADJM_6VALUE',
        'M_VALUE',
        'DAJCCSTG',
        'ADJAJCCSTG',
        'AJ_3SEER']
    for c in colunas:
        db[c] = db[c].applymap(add_digito)
    return db


# In[8]:


for i, path_base in enumerate(glob.iglob(path_base_origem + '/**/*.csv', recursive=True)):
    antigo = time.time()
    path_db_tratado = os.path.join(path_base_destino,
                               path_base.split(sep='/')[-2],
                               path_base.split(sep='/')[-1][:-4] + '.csv')
    if os.path.exists(path_db_tratado):
        continue
    if not os.path.exists(os.path.dirname(path_db_tratado)):
        os.makedirs(os.path.dirname(path_db_tratado))
    db = cudf.read_csv(path_base, ignore_index=True)
    db = tratar_1(db)
    db.to_csv(path_db_tratado, index=False)
    print('Tempo: {0} s | Salvo em: {1}'.format(round(time.time() - antigo, 2), path_db_tratado))
    del db

