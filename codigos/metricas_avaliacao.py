# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 08:20:53 2020

@author: sergio.vasconcelos
"""

from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score, roc_auc_score
from sklearn.metrics import average_precision_score, confusion_matrix
import scikitplot as skplt
import matplotlib.pyplot as plt
import numpy as np
from scikitplot.helpers import binary_ks_curve


class computar_metricas:
    def __init__(self, y_true, y_pred, classes, quiet=False,
                 y_pred_scores=None, norma_cm=True, titulo_cm=None,
                 show_num_cm=False, cmap_cm=plt.cm.Blues):
        multiclass = False
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_scores = y_pred_scores
        self.classes = classes
        self.quiet = quiet

        self.norma_cm = norma_cm
        self.titulo_cm = titulo_cm
        self.show_num_cm = show_num_cm
        self.cmap_cm = cmap_cm
        if len(self.classes) > 2:
            multiclass = True
        else:
            multiclass = False
        computar_metricas.calcular_metricas(self, multiclass=multiclass)

        if self.y_pred_scores is not None:
            computar_metricas.calcular_metricas_scores(
                self,
                multiclass=multiclass)
        else:
            self.auroc = None
            self.aupr = None
            self.ks = None
            self.ks_x = None

        self.norma_cm = False
        computar_metricas.matriz_confusao(self)
        self.norma_cm = True
        computar_metricas.matriz_confusao(self)
        computar_metricas.print_metricas(self, multiclass=multiclass)
        return

    def calcular_metricas(self, multiclass):
        if multiclass:
            opcao = None
        else:
            opcao = 'binary'
        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        self.recall = recall_score(self.y_true,
                                   self.y_pred, average=opcao)
        self.precision = precision_score(self.y_true,
                                         self.y_pred, average=opcao)
        self.f1 = f1_score(self.y_true, self.y_pred, average=opcao)
        return

    def calcular_metricas_scores(self, multiclass):
        if self.quiet is False:
            skplt.metrics.plot_roc(self.y_true, self.y_pred_scores)
            plt.show()
        if multiclass:
            self.ks = []
            self.ks_x = []
            self.auroc = []
            self.aupr = []
            for i in range(len(self.classes)):
                a = np.copy(self.y_true)
                a[a == i] = 0
                a[a != 0] = 1
                b = np.copy(self.y_pred_scores)
                c = np.copy(b[:, i])
                b[:, 1] = np.sum(np.delete(b, i, axis=1), axis=1)
                b[:, 0] = c
                if self.quiet is False:
                    skplt.metrics.plot_ks_statistic(a, b)
                    plt.title('Estatística KS para classe {0}'.format(
                        self.classes[i]))
                    plt.show()
                b = b[:, 1]
                _, _, _, ks, ks_x, _ = binary_ks_curve(a, b)

                self.ks.append(ks)
                self.ks_x.append(ks_x)
                auroc = roc_auc_score(a, b)
                aupr = average_precision_score(a, b)
                self.auroc.append(auroc)
                self.aupr.append(aupr)
            return
        if self.quiet is False:
            skplt.metrics.plot_ks_statistic(self.y_true,
                                            self.y_pred_scores)
            plt.show()
        self.y_pred_scores = self.y_pred_scores[:, 1]
        _, _, _, self.ks, self.ks_x, _ = binary_ks_curve(
            self.y_true,
            self.y_pred_scores.ravel())
        self.auroc = roc_auc_score(self.y_true, self.y_pred_scores)
        self.aupr = average_precision_score(self.y_true,
                                            self.y_pred_scores)
        return

    def print_metricas(self, multiclass):
        if multiclass:
            for i in range(len(self.classes)):
                print()
                print("Classe atual: {0}".format(self.classes[i]))
                print("{metric:<18}{value:.2f} %".format(
                    metric="Acurácia:",
                    value=self.cm.diagonal()[i] * 100))
                print("{metric:<18}{value:.2f} %".format(
                    metric="Precisão:",
                    value=self.precision[i] * 100))
                print("{metric:<18}{value:.2f} %".format(
                    metric="Revocação:",
                    value=self.recall[i] * 100))
                print("{metric:<18}{value:.2f} %".format(
                    metric="F1:",
                    value=self.f1[i] * 100))
                if self.auroc is not None:
                    print("{metric:<18}{value:.2f} %".format(
                        metric="AUROC:",
                        value=self.auroc[i] * 100))
                if self.aupr is not None:
                    print("{metric:<18}{value:.2f} %".format(
                        metric="AUPR:",
                        value=self.aupr[i] * 100))
                if self.ks is not None:
                    print(
                        "{metric:<18}{value:.2f} % no threshold de {pos:.2f} %".format(
                            metric="KS:",
                            value=self.ks[i] * 100,
                            pos=self.ks_x[i] * 100))
        else:
            print("Classe {0}:".format(self.classes[-1]))
            print("{metric:<18}{value:.2f} %".format(
                metric="Acurácia:",
                value=self.accuracy * 100))
            print("{metric:<18}{value:.2f} %".format(
                metric="Precisão:",
                value=self.precision * 100))
            print("{metric:<18}{value:.2f} %".format(
                metric="Revocação:",
                value=self.recall * 100))
            print("{metric:<18}{value:.2f} %".format(
                metric="F1:",
                value=self.f1 * 100))
            if self.auroc is not None:
                print("{metric:<18}{value:.2f} %".format(
                    metric="AUROC:",
                    value=self.auroc * 100))
            if self.aupr is not None:
                print("{metric:<18}{value:.2f} %".format(
                    metric="AUPR:",
                    value=self.aupr * 100))
            if self.ks is not None:
                print(
                    "{metric:<18}{value:.2f} % no threshold de {pos:.2f} %".format(
                        metric="KS:",
                        value=self.ks * 100,
                        pos=self.ks_x * 100))
        return

    def matriz_confusao(self):
        if self.titulo_cm is None:
            if self.norma_cm:
                self.titulo_cm = 'Matriz de confusão normalizada'
            else:
                self.titulo_cm = 'Matriz de confusão'

        self.cm = confusion_matrix(self.y_true, self.y_pred)
        if self.norma_cm is True:
            self.cm = self.cm.astype('float') / self.cm.sum(
                    axis=1)[:, np.newaxis]
        if self.show_num_cm is True:
            if self.norma_cm is True:
                print('Matriz de confusão normalizada')
            else:
                print('Matriz de confusão')
            print(self.cm)

        fig, ax = plt.subplots()
        im = ax.imshow(self.cm, interpolation='nearest', cmap=self.cmap_cm)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(self.cm.shape[1]),
               yticks=np.arange(self.cm.shape[0]),
               xticklabels=self.classes, yticklabels=self.classes,
               title=self.titulo_cm,
               ylabel='Label verdadeira',
               xlabel='Label prevista')

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        fmt = '.2f' if self.norma_cm else 'd'
        thresh = self.cm.max() / 2.
        for i in range(self.cm.shape[0]):
            for j in range(self.cm.shape[1]):
                ax.text(j, i, format(self.cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if self.cm[i, j] > thresh else "black")
        fig.tight_layout()
        if self.quiet is False:
            plt.show()
        return
