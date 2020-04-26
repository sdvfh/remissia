import os
import time
import glob
import cudf
import pandas as pd
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 100)


# %%
def ler_dicionario(dic_variaveis):
    '''Funcao que recebe o dicionario em dataframe que retorna:
    1. lista com o nome, a posicao e o tamanho das variaveis;
    2. lista com o nome das variaveis.'''
    lista = []
    for i in range(len(dic_variaveis)):

        tam_var = dic_variaveis.iloc[i].to_string()[41:43].strip()
        if tam_var[-1] == '.':
            tam_var = tam_var[0]
        tam_var = int(tam_var)

        nome_var = dic_variaveis.iloc[i].to_string()[15:35].strip()
        pos_var = int(dic_variaveis.iloc[i].to_string()[11:14].strip())

        lista.append([nome_var, pos_var, tam_var])
    lista_nomes = [nome for nome, pos, tam in lista]
    return lista, lista_nomes


# %%
dir_base_origem = 'SEER_1975_2016_CUSTOM_TEXTDATA'
dir_base_destino = 'base_col_sep_quimo'
dic_variaveis = 'dicionario_dados_quimo.txt'

# %%
path_base_origem = '../{0}/incidence'.format(dir_base_origem)
path_base_destino = '../{0}'.format(dir_base_destino)
dic_variaveis = pd.read_csv('../dicionarios/{0}'.format(dic_variaveis),
                            header=None)
lista, lista_nomes = ler_dicionario(dic_variaveis)

# %%
for base_path in glob.iglob('{0}/**/*.TXT'.format(path_base_origem),
                            recursive=True):
    antigo = time.time()
    path_base_destino_temp = os.path.join(
        dir_base_destino,
        base_path.split(sep='/')[-2],
        base_path.split(sep='/')[-1][:-4] + '.csv')

    if not os.path.exists(os.path.dirname(path_base_destino_temp)):
        os.makedirs(os.path.dirname(path_base_destino_temp))
    df = cudf.read_csv(base_path, header=None)

    for nome, pos, tam in lista:
        pos -= 1
        df[nome] = df['0'].str.slice(pos, pos + tam).str.strip()
    df = df.drop('0', axis=1)
    df.to_csv(path_base_destino_temp, index=False)

    print('Tempo: {0} s | Salvo em: {1}'.format(
        round(time.time() - antigo, 2),
        path_base_destino_temp))
