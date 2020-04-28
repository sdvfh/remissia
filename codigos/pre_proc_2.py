import os
import time
import glob
import pickle
import cudf
import pandas as pd
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)
cudf.set_allocator('managed', pool=True)


# %%
def carregar_lista(coluna):
    '''Carrega a lista de correspondencia em numeros da respectiva tabela.'''
    with open('../dicionarios/listas/{0}.pkl'.format(coluna), 'rb') as arq:
        lista = pickle.load(arq)
    return lista


def tratar_adicional(db, coluna):
    '''Tratamento adicional requerido por alguma das tabelas.'''
    if coluna == 'DASRCN':
        db.loc[db[coluna] == '10-', coluna] = '10'
        db.loc[db[coluna] == '10+', coluna] = '10'
        db.loc[db[coluna] == '20+', coluna] = '20'
        db.loc[db[coluna] == '30-', coluna] = '30'
        db.loc[db[coluna] == '30+', coluna] = '30'
        db.loc[db[coluna] == '40+', coluna] = '40'

    elif coluna == 'DASRCM':
        db.loc[db[coluna] == '10+', coluna] = '10'

    elif coluna == 'DSRPSG':
        db.loc[db[coluna].str.strip() == '', coluna] = None
        db[coluna] = db[coluna].astype(float)
        values = [
            [4, 700],
            [3, 500],
            [1, 100],
            [2, 300],
            ]
        for orig, mod in values:
            db.loc[db[coluna] == orig, coluna] = mod

    return db


def aplicar_dicionario(db):
    '''Trata as colunas a depender das informacoes no dicionario,
    principalmente a substituicao de nulos para NaN e exclusao de
    colunas selecionadas.'''
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

        if all(db[nome][db[nome].notnull()].str.isdigit()) or (
                db[nome][db[nome] != ''].shape[0] == 0):
            db[nome] = db[nome].astype(float)
    return db


def add_digito(num):
    '''Adiciona o digito adicional a depender do numero na coluna.'''
    if num % 10 == 9:
        return num * 10 + 9
    elif num == 88:
        return num * 10 + 8
    else:
        return num * 10


def apply_add_digito(db):
    '''Aplicacao do digito adicional nas colunas selecionadas.'''
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


def tratar_principal(db):
    '''Parte 2 do pre-processamento da tabela.'''
    colunas_tratar = ['DASRCT',
                      'DASRCN',
                      'DASRCM',
                      'DSRPSG']
    for coluna in colunas_tratar:
        lista = carregar_lista(coluna)
        for orig, mod in lista:
            db[coluna] = db[coluna].str.replace(orig, mod)
        db = tratar_adicional(db, coluna)

    db = aplicar_dicionario(db)

    db = db.drop(columns=['eod13_sz',
                          'eod13_psv',
                          'eod13_ex',
                          'eod13_ss',
                          'eod13_nd',
                          'eod13_dnd',
                          'eod13_dss',
                          'EOD2',
                          'eod4_sz',
                          'eod4_extent',
                          'edo4_nd'])

    db = apply_add_digito(db)
    return db


# %%
dir_base_origem = 'base_col_sep_quimo'
dir_base_destino = 'base_col_trat1_quimo'
dic_variaveis = 'variaveis_quimo.csv'

# %%
dic_variaveis = pd.read_csv('../dicionarios/{0}'.format(dic_variaveis),
                            sep=';')

# %%
for i, path_base in enumerate(glob.iglob(
        '../bases/{0}/**/*.csv'.format(dir_base_origem),
        recursive=True)):

    antigo = time.time()
    path_base_tratada = '../bases/{0}/{1}/{2}.csv'.format(
        dir_base_destino,
        path_base.split(sep='/')[-2],
        path_base.split(sep='/')[-1][:-4])

    if os.path.exists(path_base_tratada):
        continue
    if not os.path.exists(os.path.dirname(path_base_tratada)):
        os.makedirs(os.path.dirname(path_base_tratada))

    db = cudf.read_csv(path_base, ignore_index=True)
    db = tratar_principal(db)
    db.to_csv(path_base_tratada, index=False)

    print('Tempo: {0} s | Salvo em: {1}'.format(
        round(time.time() - antigo, 2),
        path_base_tratada))
    del db
