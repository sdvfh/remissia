from base_seer_custom import Base_seer

def criar_select(lista, nome_coluna):
    string_final = 'CASE '
    for to_replace, value in lista_T:
        string_final += 'WHEN {0} = \'{1}\' THEN \'{2}\'\n'.format(nome_coluna, to_replace, value)
    string_final += 'END'
    return string_final

def add_digito(nome_coluna):
    return '''
    CASE
     WHEN {0} % 10 = 9 THEN ({0} * 10) + 9
     WHEN {0} = 88 then ({0} * 10) + 8
     ELSE {0} * 10 END'''.format(nome_coluna)

s = Base_seer('../bases/SEER_1975_2016_CUSTOM_TEXTDATA')


s.df = s.df.fillna(0, subset='SRV_TIME_MON')

lista_T = [
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

DASRCT = criar_select(lista_T, 'DASRCT')

DAJCCT, ADJTM_6VALUE, T_VALUE = add_digito('DAJCCT'), add_digito('ADJTM_6VALUE'), add_digito('T_VALUE')

lista_N = [
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

DASRCN = criar_select(lista_N, 'DASRCN')

DAJCCN, ADJNM_6VALUE, N_VALUE = add_digito('DAJCCN'), add_digito('ADJNM_6VALUE'), add_digito('N_VALUE')

lista_M = [
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

DASRCM = criar_select(lista_M, 'DASRCM')

DAJCCM, ADJM_6VALUE, M_VALUE = add_digito('DAJCCM'), add_digito('ADJM_6VALUE'), add_digito('M_VALUE')

lista_stage = [
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

DSRPSG = criar_select(lista_stage, 'DASRCM')

DAJCCSTG, ADJAJCCSTG, AJCC_STG = add_digito('DAJCCSTG'), add_digito('ADJAJCCSTG'), add_digito('AJCC_STG')

# Commented out IPython magic to ensure Python compatibility.
# %cd ../bases

query = '''
SELECT 
--    ######################################## TARGETS #######################################
    
       CASE
         WHEN SRV_TIME_MON >= 60 THEN 1 ELSE 0 END      AS CURADO
     , SEER.SRV_TIME_MON                                AS TEMPO_SOBRE
     , CASE
         WHEN VSRTSADX = 1 THEN 1
         WHEN VSRTSADX IN (0, 8) THEN 0
         ELSE NULL END                                  AS SOBRE_COM_CANCER
         
--    ################################### DADOS DO PACIENTE #######################################
     
     , REC_NO - 1                                       AS HISTORICO_TUMOR
     , YR_BRTH                                          AS ANO_NASC
     , SEER.MDXRECMP                                    AS MES_DIAG
     , SEER.YEAR_DX                                     AS ANO_DIAG
     , AGE_DX                                           AS IDADE_DIAG
     , CAST(substring(ST_CNTY, 0, 2) AS INTEGER)        AS ESTADO
     , CAST(substring(ST_CNTY, 3, 3) AS INTEGER)        AS CIDADE
     , RACE1V                                           AS ETNIA
     , SEX                                              AS SEXO
     , MAR_STAT                                         AS ESTADO_CIVIL
     , INSREC_PUB                                       AS PLANO_SAUDE
     
--     ################################### CODIGOS DO TUMOR #######################################
     
     , PRIMSITE                                         AS CID
     , SITERWHO                                         AS CID_SEER
     , ICCC3XWHO                                        AS ICCC
     , AYASITERWHO                                      AS AYA
     , HISTO3V                                          AS HIST
     , BEHO3V                                           AS HIST_COMP
     , HISTREC                                          AS HIST_SEER
     , HISTRECB                                         AS HIST_SEER_CEREBRO

--     ############################## ESPECIFICAÇÕES DO TUMOR ####################################### 
 
     , TIPO_TUMOR                                       AS TIPO_TUMOR
     , FIRSTPRM                                         AS PRI_TUMOR
     , COALESCE(
         DAJCC7T
       , {query_DAJCCT}
       , {query_ADJTM_6VALUE}
       , {query_T_VALUE}
       , CAST({query_DASRCT} AS INTEGER))               AS CLASS_T
     , COALESCE(
         DAJCC7N
       , {query_DAJCCN}
       , {query_ADJNM_6VALUE}
       , {query_N_VALUE}
       , CAST({query_DASRCN} AS INTEGER))               AS CLASS_N
     , COALESCE(
         DAJCC7M
       , {query_DAJCCM}
       , {query_ADJM_6VALUE}
       , {query_M_VALUE}
       , CAST({query_DASRCM} AS INTEGER))               AS CLASS_M
     , COALESCE(
         SCSSM2KO
       , DSS1977S
       , SSS77VZ
       , SSSM2KPZ
       , SUMM2K
       , CASE
           WHEN HST_STGA = 4 THEN 7 ELSE HST_STGA END)  AS ESTAGIO_SUMARIZADO
     , COALESCE(
         DAJCC7STG
       , {query_DAJCCSTG} 
       , {query_ADJAJCCSTG} 
       , CAST({query_DSRPSG} AS INTEGER) 
       , CASE
           WHEN ANNARBOR = 1 THEN 100
           WHEN ANNARBOR = 2 THEN 300
           WHEN ANNARBOR = 3 THEN 500
           WHEN ANNARBOR = 4 THEN 700 END 
       , CASE WHEN {query_AJCC_STG} = 980 THEN 888 ELSE {query_AJCC_STG} END
                                                      ) AS ESTAGIO
       , LATERAL                                        AS LATERALIDADE
     , CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END
                                                        AS MALIGNO
     , CASE WHEN (CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END = 1) THEN (ROW_NUMBER() OVER (PARTITION BY PUBCSNUM, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END ORDER BY REC_NO) - 1) ELSE (REC_NO - ROW_NUMBER() OVER (PARTITION BY PUBCSNUM, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END ORDER BY REC_NO, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END)) END + CASE WHEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) < 60 THEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) - CASE WHEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) = 0 THEN 0 ELSE 1 END ELSE 0 END
                                                        AS HISTORICO_MALIGNO
     , CASE WHEN (CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END = 1) THEN (REC_NO - ROW_NUMBER() OVER (PARTITION BY PUBCSNUM, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END ORDER BY REC_NO, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END)) ELSE (ROW_NUMBER() OVER (PARTITION BY PUBCSNUM, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END ORDER BY REC_NO) - 1) END + CASE WHEN (first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) >= 60) AND (first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) NOT IN (88, 99)) THEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) - 60 - CASE WHEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) = 60 THEN 0 ELSE 1 END ELSE 0 END
                                                        AS HISTORICO_BENIGNO

--     ############################## INFORMAÇÕES DE METÁSTASE ######################################  

     , SCMETSDXB_PUB                                    AS MET_OSSO
     , SCMETSDXBR_PUB                                   AS MET_CEREBRO
     , SCMETSDXLIV_PUB                                  AS MET_FIGADO
     , SCMETSDXLUNG_PUB                                 AS MET_PULMAO
     
--     ###################################### LINFONODOS ############################################      

     , CASE
         WHEN EOD10_NE >= 90 THEN NULL ELSE EOD10_NE END AS LINF_EXA
     , CASE
         WHEN EOD10_PN >= 90 THEN NULL ELSE EOD10_PN END AS LINF_EXA_POS

--     ###################################### CIRURGIA ############################################## 

     , NO_SURG                                           AS NAO_CIRURGIA
     , SURGSCOF                                          AS CIRURGIA_LINF
     , SURGSITF                                          AS CIRURGIA_OUTRO

--     ###################################### RADIOTERAPIA ########################################## 
   
     , CHEMO_RX_REC                                      AS QUIMIOTERAPIA
     , RADIATNR                                          AS RADIOTERAPIA
     , RAD_SURG                                          AS SEQ_TRATAMENTO

FROM SEER

'''.format(query_DAJCCT=DAJCCT, query_ADJTM_6VALUE=ADJTM_6VALUE,
           query_T_VALUE=T_VALUE, query_DASRCT=DASRCT,
           query_DAJCCN=DAJCCN, query_ADJNM_6VALUE=ADJNM_6VALUE,
           query_N_VALUE=N_VALUE, query_DASRCN=DASRCN,
           query_DAJCCM=DAJCCM, query_ADJM_6VALUE=ADJM_6VALUE,
           query_M_VALUE=M_VALUE, query_DASRCM=DASRCM, 
           query_DAJCCSTG=DAJCCSTG, query_ADJAJCCSTG=ADJAJCCSTG, 
           query_AJCC_STG=AJCC_STG, query_DSRPSG=DSRPSG)
s.spark.sql(query).coalesce(1).write.csv('base_seer_customizada',header=True)

# Commented out IPython magic to ensure Python compatibility.
# %pwd

# TESTES
query = '''
SELECT PUBCSNUM
     , REC_NO
     , SEQ_NUM
     , CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END AS MALIGNO
     , CASE
         WHEN SEQ_NUM = 0 THEN 0
         WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN SEQ_NUM - 1
         WHEN REC_NO = 1 THEN 0 
         END AS HISTORICO_MALIGNO
    , ROW_NUMBER() OVER (PARTITION BY PUBCSNUM  ) AS TESTE1
from seer
WHERE CASE
         WHEN SEQ_NUM = 0 THEN 0
         WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN SEQ_NUM - 1
         WHEN REC_NO = 1 THEN 0 END IS NULL
ORDER BY PUBCSNUM, REC_NO, SEQ_NUM
limit 100
'''.format()
s.spark.sql(query)

query = '''
SELECT PUBCSNUM
     , REC_NO
     , SEQ_NUM
     , CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END AS MALIGNO
     , CASE WHEN (CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END = 1) THEN (ROW_NUMBER() OVER (PARTITION BY PUBCSNUM, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END ORDER BY REC_NO) - 1) ELSE (REC_NO - ROW_NUMBER() OVER (PARTITION BY PUBCSNUM, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END ORDER BY REC_NO, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END)) END AS HISTORICO_MALIGNO
     , CASE WHEN (CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END = 1) THEN (REC_NO - ROW_NUMBER() OVER (PARTITION BY PUBCSNUM, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END ORDER BY REC_NO, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END)) ELSE (ROW_NUMBER() OVER (PARTITION BY PUBCSNUM, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END ORDER BY REC_NO) - 1) END AS HISTORICO_BENIGNO
     , CASE WHEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM) < 60 THEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM) END AS PRIMEIRO_VALOR
from seer
WHERE PUBCSNUM = 23
ORDER BY PUBCSNUM, REC_NO
limit 100
'''.format()
s.spark.sql(query).show(100)

query = '''
SELECT PUBCSNUM
     , REC_NO
     , SEQ_NUM
     , CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END AS MALIGNO
     , CASE WHEN (CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END = 1) THEN (ROW_NUMBER() OVER (PARTITION BY PUBCSNUM, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END ORDER BY REC_NO) - 1) ELSE (REC_NO - ROW_NUMBER() OVER (PARTITION BY PUBCSNUM, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END ORDER BY REC_NO, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END)) END + CASE WHEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) < 60 THEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) - 1 ELSE 0 END AS HISTORICO_MALIGNO
     , CASE WHEN (CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END = 1) THEN (REC_NO - ROW_NUMBER() OVER (PARTITION BY PUBCSNUM, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END ORDER BY REC_NO, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END)) ELSE (ROW_NUMBER() OVER (PARTITION BY PUBCSNUM, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END ORDER BY REC_NO) - 1) END + CASE WHEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) >= 60 THEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) - 60 - 1 ELSE 0 END AS HISTORICO_BENIGNO
     , CASE WHEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) < 60 THEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) ELSE 0 END AS PRIMEIRO_VALOR
     , first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) as teste
from seer
WHERE PUBCSNUM = 69960793
ORDER BY PUBCSNUM, REC_NO
limit 100
'''.format()
s.spark.sql(query).show(100)

query = '''
SELECT PUBCSNUM
     , REC_NO
     , SEQ_NUM
     , CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END AS MALIGNO
     , CASE WHEN (CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END = 1) THEN (ROW_NUMBER() OVER (PARTITION BY PUBCSNUM, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END ORDER BY REC_NO) - 1) ELSE (REC_NO - ROW_NUMBER() OVER (PARTITION BY PUBCSNUM, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END ORDER BY REC_NO, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END)) END + CASE WHEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) < 60 THEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) - 1 ELSE 0 END AS HISTORICO_MALIGNO
     , CASE WHEN (CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END = 1) THEN (REC_NO - ROW_NUMBER() OVER (PARTITION BY PUBCSNUM, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END ORDER BY REC_NO, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END)) ELSE (ROW_NUMBER() OVER (PARTITION BY PUBCSNUM, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END ORDER BY REC_NO) - 1) END + CASE WHEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) >= 60 THEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) - 60 - 1 ELSE 0 END AS HISTORICO_BENIGNO
--     , CASE WHEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) < 60 THEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) ELSE 0 END AS PRIMEIRO_VALOR
     , first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) as PRIMEIRO_VALOR
from seer
WHERE PUBCSNUM = 29438717
ORDER BY PUBCSNUM, REC_NO
limit 100
'''.format()
s.spark.sql(query).show(100)

query = '''
SELECT PUBCSNUM
     , REC_NO
     , SEQ_NUM
     , CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END AS MALIGNO
     , CASE WHEN (CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END = 1) THEN (ROW_NUMBER() OVER (PARTITION BY PUBCSNUM, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END ORDER BY REC_NO) - 1) ELSE (REC_NO - ROW_NUMBER() OVER (PARTITION BY PUBCSNUM, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END ORDER BY REC_NO, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END)) END + CASE WHEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) < 60 THEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) - CASE WHEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) = 0 THEN 0 ELSE 1 END ELSE 0 END AS HISTORICO_MALIGNO
     , CASE WHEN (CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END = 1) THEN (REC_NO - ROW_NUMBER() OVER (PARTITION BY PUBCSNUM, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END ORDER BY REC_NO, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END)) ELSE (ROW_NUMBER() OVER (PARTITION BY PUBCSNUM, CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END ORDER BY REC_NO) - 1) END + CASE WHEN (first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) >= 60) AND (first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) NOT IN (88, 99)) THEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) - 60 - CASE WHEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) = 60 THEN 0 ELSE 1 END ELSE 0 END AS HISTORICO_BENIGNO
--     , CASE WHEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) < 60 THEN first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) ELSE 0 END AS PRIMEIRO_VALOR
--     , first_value(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) as PRIMEIRO_VALOR
from seer
WHERE PUBCSNUM = 148
ORDER BY PUBCSNUM, REC_NO
limit 100
'''.format()
s.spark.sql(query).show(100)

query = '''
select *
    from
    (
        SELECT PUBCSNUM
             , FIRST_VALUE(SEQ_NUM) OVER (PARTITION BY PUBCSNUM ORDER BY REC_NO) AS PRIMEIRO_VALOR
        from seer
    ) teste
    where teste.primeiro_valor = 0
    limit 100
'''.format()
s.spark.sql(query).show(100)

query = '''
SELECT PUBCSNUM
     , SUM(CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END) AS MALIGNO
     , SUM(CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 0 ELSE 1 END) AS BENIGNO

from seer
group by PUBCSNUM
ORDER BY (SUM(CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 0 ELSE 1 END) * 0.7 + SUM(CASE WHEN SEQ_NUM < 60 OR SEQ_NUM = 99 THEN 1 ELSE 0 END) * 0.3) DESC
limit 100
'''.format()
s.spark.sql(query).show(100)