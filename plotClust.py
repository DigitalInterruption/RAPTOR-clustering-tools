import numpy as np

import matplotlib.pyplot as plt

# function to plot multi-variable cluster data with centrodis
def multiVar(Y, labels, centroids, classes):
    nDims = Y.shape[1]
    dimNames = Y.columns
    nPlots = nDims ** 2
    nClasses = np.arange(0, len(classes))
    colours = plt.get_cmap('tab10').colors
    empties = [n * (nDims + 1) for n in range(nDims)]
    fig = plt.figure()
    axes = [fig.add_subplot(nDims, nDims, n+1) for n in range(nPlots)]
    x, y = 0, 0
    for n in range(nPlots):
        if n in empties:
            axes[n].text(0.1, 0.1,
                    '\n'.join(dimNames[int(n / (nDims + 1))].split('-')),
                    wrap=True, multialignment='center')
        else:
            for l in nClasses:
                try: axes[n].scatter(
                        Y[dimNames[x]][labels == l],
                        Y[dimNames[y]][labels == l],
                        s=1, cmap=colours[l], label=classes[l], alpha=.85, marker='.')
                except: print('no data to plot for '
                        + classes[l] + ', in graph '
                        + dimNames[x] + ' vs. ' + dimNames[y])
            axes[n].scatter(
                    centroids[:,x],
                    centroids[:,y],
                    s=10, marker='X', color='black')
        axes[n].set_xticks([])
        axes[n].set_yticks([])
        x += 1
        if x == nDims:
            x = 0
            y += 1
    plt.figlegend(labels = classes, markerscale=5)
    plt.show()
