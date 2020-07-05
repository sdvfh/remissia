# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 22:12:12 2020

@author: sergi
"""

import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score, roc_auc_score
from sklearn.metrics import average_precision_score, confusion_matrix
from scikitplot.helpers import binary_ks_curve

# TODO: métricas de desempenho para multilabel

class Metricas():
    def __init__(self, y_true, y_pred, y_pred_scores=None,
                 nm_classes=None, show_fig=True):
        self.y_true = y_true
        # TODO: análise multi-task
        if len(self.y_true.shape) > 1:
            raise NotImplementedError
        self.y_pred = y_pred
        self.nm_classes = np.unique(y_true) if nm_classes is None \
            else nm_classes
        self.show_fig = show_fig
        self.y_pred_scores = y_pred_scores
        self.multi_label = len(np.unique(self.y_true)) > 2
        Metricas.calculo(self)
        Metricas.print_result_classe(self)
        return

    def calculo(self):
        Metricas.desempenho(self)
        if self.show_fig:
            Metricas.graficos(self)
        return
    
    def desempenho(self):
        self.acuracia = Metricas.calc_acuracia(self)
        self.sensibilidade = Metricas.calc_sensibilidade(self)
        self.precisao = Metricas.calc_precisao(self)
        self.f1 = Metricas.calc_f1(self)
        self.auroc = Metricas.calc_auroc(self)
        self.ks, self.ks_x = Metricas.calc_ks(self)
        return

    def calc_acuracia(self):
        acuracia = accuracy_score(self.y_true, self.y_pred)
        return acuracia

    def calc_sensibilidade(self):
        sensibilidade = recall_score(self.y_true, self.y_pred)
        return sensibilidade

    def calc_precisao(self):
        precisao = precision_score(self.y_true, self.y_pred)
        return precisao

    def calc_f1(self):
        f1 = f1_score(self.y_true, self.y_pred)
        return f1

    def calc_auroc(self):
        auroc = roc_auc_score(self.y_true, self.y_pred_scores[: , 1])
        return auroc

    def calc_ks(self):
        if (self.multi_label is False) and (self.y_pred_scores is not None):
            _, _, _, ks, ks_x, _ = binary_ks_curve(
                self.y_true,
                self.y_pred_scores[:, 1].ravel())  
            return ks, ks_x
        else:
            return None, None

    def graficos(self):
        Metricas.matriz_confusao(self)
        Metricas.ks_stats(self)
        Metricas.roc(self)
        return

    def matriz_confusao(self):
        titulo_cm = 'Matriz de confusão normalizada'
        self.matriz_confusao = confusion_matrix(self.y_true, self.y_pred)
    
        self.matriz_confusao = (
            self.matriz_confusao.astype('float')
            / self.matriz_confusao.sum(axis=1)[:, np.newaxis])
    
        print('Matriz de confusão normalizada')
        print(self.matriz_confusao)

        fig, ax = plt.subplots()
        im = ax.imshow(self.matriz_confusao, interpolation='nearest',
                       cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(self.matriz_confusao.shape[1]),
               yticks=np.arange(self.matriz_confusao.shape[0]),
               xticklabels=self.nm_classes, yticklabels=self.nm_classes,
               title=titulo_cm,
               ylabel='Label true',
               xlabel='Label predicted')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        
        fmt = '.2f' # if self.norma_cm else 'd'
        thresh = self.matriz_confusao.max() / 2.
        for i in range(self.matriz_confusao.shape[0]):
            for j in range(self.matriz_confusao.shape[1]):
                ax.text(j, i, format(self.matriz_confusao[i, j], fmt),
                        ha="center", va="center",
                        color="white" if self.matriz_confusao[i, j] > thresh \
                            else "black")
        fig.tight_layout()
        plt.show()
        return

    def ks_stats(self):
        skplt.metrics.plot_ks_statistic(self.y_true, self.y_pred_scores)
        plt.show()
        return

    def roc(self):
        skplt.metrics.plot_roc(self.y_true, self.y_pred_scores,
                               plot_micro=False, plot_macro=False,
                               classes_to_plot=[1])
        plt.show()
        return

    def print_result_classe(self):
        print("Classe {0}:".format(self.nm_classes[-1]))
        print("{metric:<15}{value:.2f} %".format(
            metric="Acurácia:", value=self.acuracia * 100))
        print("{metric:<15}{value:.2f} %".format(
            metric="Precisão:", value=self.precisao * 100))
        print("{metric:<15}{value:.2f} %".format(
            metric="Sensibilidade:",
            value=self.sensibilidade * 100))
        print("{metric:<15}{value:.2f} %".format(
            metric="F1:", value=self.f1 * 100))
        if self.auroc is not None:
            print("{metric:<15}{value:.2f} %".format(
                metric="AUROC:", value=self.auroc * 100))
        if self.ks is not None:
            print(
                "{metric:<15}{value:.2f} % no threshold de {pos:.2f} %".format(
                    metric="KS:", value=self.ks * 100, pos=self.ks_x * 100))
        return
