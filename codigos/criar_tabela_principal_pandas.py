import pandas as pd
import glob
import numpy as np

path = r"C:\arquivos_spark\SEER_1975_2016_CUSTOM_TEXTDATA\incidence"
paths = glob.glob(path + "\**\*.txt", recursive=True)

dicionario = pd.read_csv(r'D:\repositorios\remissia\dicionarios\carga_inicial.csv', sep=';')

dfs = []
for path in paths:
    df = pd.read_csv(path, header=None)
    for variavel in dicionario.iterrows():
        nome = variavel[1]['Name']
        pos = variavel[1]['Position']
        tam = variavel[1]['Length']
        tipo = variavel[1]['Type']
        valor_nulo = variavel[1]['Null if']
        df[nome] = df[0].str.slice(start=pos, stop=pos+tam).str.strip()
        if not np.isnan(valor_nulo):
            df[nome] = df[nome].str.replace(str(valor_nulo), '')
        if tipo == 'Integer':
            df[nome] = pd.to_numeric(df[nome], downcast='unsigned', errors='coerce')
    df.drop(columns=0, inplace=True)
    dfs.append(df)

df = pd.concat(dfs)
del dfs

# %%
df.to_csv(r'C:\arquivos_spark\seer_bruto.csv', index=False)