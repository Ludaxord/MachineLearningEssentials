# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from .regressor import Regressor

class Visualisator:
    
    def visualiseColoredMap(self, X_set, y_set, colors, title, xLabel, yLabel, classifier = None, withSaving = True, gridVisible = True):
        scatterColor = colors[0]
        plotColor = colors[1]
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        if classifier is None:
            classifier = Regressor.LogisticRegressionModelCreator(X_set, y_set)
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                         alpha = 0.75, cmap = ListedColormap((scatterColor, plotColor)))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap((scatterColor, plotColor))(i), label = j)

        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.grid(gridVisible)
        plt.legend()
        if withSaving == True:
            plt.savefig(f'RegressionFigures/{title}.png', dpi = 300)
        plt.show()
    
    def visualisePrediction(self, xScatter, scatterLabel, yVector, xPlot, plotLabel, colors, title, xLabel, yLabel, withSaving = True, gridVisible = True, anotherXPlot = None):
        scatterColor = colors[0]
        plotColor = colors[1]
        plt.scatter(xScatter, yVector, color = scatterColor, label = scatterLabel)
        if anotherXPlot is None:
            plt.plot(xScatter, xPlot, color = plotColor, label = plotLabel)
        else:
            plt.plot(anotherXPlot, xPlot, color = plotColor, label = plotLabel)

        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.grid(gridVisible)
        plt.legend()
        if withSaving == True:
            plt.savefig(f'RegressionFigures/{title}.png', dpi = 300)
        plt.show()
        
    def visualisePredictions(self, xScatter, scatterLabel, yVector, plots, labels, mainColor, colors, title, xLabel, yLabel, withSaving = True, gridVisible = True):
        plt.scatter(xScatter, yVector, color = mainColor, label = scatterLabel)
        for index, plot in enumerate(plots):
            plt.plot(xScatter, plot, color = colors[index], label = labels[index])
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.grid(gridVisible)
        plt.legend()
        if withSaving == True:
            plt.savefig(f'RegressionFIgures/{title}.png', dpi = 300)
        plt.show()