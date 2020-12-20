import pandas as pd
import glob
import numpy as np
from joblib import Parallel, delayed

path = r"C:\arquivos_spark\SEER_1975_2016_CUSTOM_TEXTDATA\incidence"
paths = glob.glob(path + "\**\*.txt", recursive=True)

dicionario = pd.read_csv(r'D:\repositorios\remissia\dicionarios\carga_inicial.csv', sep=';')

def tratar_base(path):
    df = pd.read_csv(path, header=None)
    tipo_tumor = path.split('\\')[-1].split('.')[0]
    pasta_origem = path.split('\\')[-2]
    df['TIPO_TUMOR'] = tipo_tumor
    df['PASTA_ORIGEM'] = pasta_origem
    for variavel in dicionario.iterrows():
        nome = variavel[1]['Name']
        pos = variavel[1]['Position']
        tam = variavel[1]['Length']
        tipo = variavel[1]['Type']
        valor_nulo = variavel[1]['Null if']
        print(pasta_origem, tipo_tumor, nome)
        df[nome] = df[0].str.slice(start=pos, stop=pos + tam).str.strip()
        if not np.isnan(valor_nulo):
            df[nome] = df[nome].str.replace(str(valor_nulo), '')
        if tipo == 'Integer':
            df[nome] = pd.to_numeric(df[nome], downcast='unsigned', errors='coerce')
    df.drop(columns=0, inplace=True)
    return df

dfs = Parallel(n_jobs=4, verbose=1, max_nbytes=None, prefer='threads')(delayed(tratar_base)(path) for path in paths)
df = pd.concat(dfs)
del dfs

# %%
df.to_csv(r'C:\arquivos_spark\seer_bruto.csv', index=False)