import sys
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

class Base_seer():
    def __init__(self, path):
        self.path = path
        self.spark = SparkSession.builder.appName("seer_pre_prop").getOrCreate()
        self.dic_dados = pd.read_csv('{0}/carga_inicial.csv'.format(self.path), sep=';')
        Base_seer.criar_tabela(self)
        self.df.createOrReplaceTempView("seer")
        print('Tabela carregada.')
        return
    
    def criar_tabela(self):
        Base_seer.ler_tabela(self)
        Base_seer.separar_colunas(self)
        Base_seer.subs_nulos(self)
        Base_seer.info_arq(self)
        return
    
    def ler_tabela(self):
        self.df = self.spark.read.csv('{0}/incidence/*'.format(self.path), pathGlobFilter='*.TXT')
        self.df = self.df.withColumnRenamed('_c0', 'linha')
        return
    
    def separar_colunas(self):
        colunas = []
        for i in range(self.dic_dados.shape[0]):
            linha = self.dic_dados.iloc[i, :]
            nome_coluna = linha[0]
            tipo_coluna = linha[1]
            pos_coluna = int(linha[3])
            if i > 0:
                pos_coluna += 1
            tam_coluna = int(linha[4])
            colunas.append(F.trim(F.substring(self.df['linha'], pos_coluna, tam_coluna)).cast(tipo_coluna).alias(nome_coluna))
        self.df = self.df.select(colunas)
        return

    def subs_nulos(self):
        self.df = self.df.replace('', None)
        for i in range(self.dic_dados.shape[0]):
            linha = self.dic_dados.iloc[i, :]
            nome_coluna = linha[0]
            try:
                rep_nulo = int(linha[-4])
            except ValueError:
                continue
            self.df = self.df.replace(rep_nulo, None, subset=nome_coluna)
        self.df = self.df.replace('9999', None, subset='ICDOT10V')
        return

    def info_arq(self):
        tipo_tumor = F.split(F.split(F.substring(F.input_file_name(), -68, 100), '/').getItem(3), '.TXT').getItem(0)
        base_ano = F.split(F.substring(F.input_file_name(), -68, 100), '/').getItem(2)
        self.df = self.df.withColumn('TIPO_TUMOR', tipo_tumor).withColumn('BASE_ANO', base_ano)
        return