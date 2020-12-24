import pandas as pd
from automl import Auto
df = pd.read_csv(r'C:\arquivos_spark\seer\base_seer_final.csv')
# TODO: analisar os valores nulos na coluna TEMPO_SOBRE
# df = df.drop(columns=['TEMPO_SOBRE', 'MORTE_CANCER'])
df = df.drop(columns=['MORTO', 'MORTE_CANCER'])
df = df[df['TIPO_TUMOR'] == 'RESPIR']

df.drop(inplace=True, columns=['CID_SEER', 'ICCC', 'AYA', 'HIST_COMP', 'HIST_SEER', 'HIST_SEER_CEREBRO',
                               'TIPO_TUMOR', 'PASTA_ORIGEM'])

continuas = ['HISTORICO_TUMOR', 'IDADE_DIAG', 'HISTORICO_MALIGNO', 'HISTORICO_BENIGNO', 'LINF_EXA_POS', 'LINF_EXA']

binarias = ['MALIGNO', 'PRI_TUMOR']

target_atual = 'TEMPO_SOBRE'
df.dropna(subset=[target_atual], inplace=True)
tipo_problema_atual = 'Regressao'

automl = Auto(nome='respiratorio_tempo_sobre', debug=1)

automl.base(df=df, target=target_atual, tipo_problema=tipo_problema_atual,
            v_continuas=continuas, v_binarias=binarias)

automl.treinar(salvar_modelo_final=False, n_jobs=-1, pontos=1, hiper_base=False)

resultado = automl.resultado