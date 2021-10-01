import numpy as np

from itertools import permutations

from scipy.stats import mode

# elect sample labels from mode ofpredictions of labels for sample data points
def electLabels(predictedLabels, trueLabels, sampleBounds):
    pLabels, tLabels, pConf = [], [], []
    for bounds in sampleBounds:
        p = mode(predictedLabels[bounds[0]:bounds[1]])
        t = mode(trueLabels[bounds[0]:bounds[1]])
        certainty = p[1][0] / (bounds[1] - bounds[0])
        pLabels.append(p[0][0])
        tLabels.append(t[0][0])
        pConf.append(certainty)

    return np.array(pLabels), np.array(tLabels), pConf


# remap cluster labels by asessing which permutation gives the highest accuracy
def remapLabels(pLabels, tLabels):
    clusters = np.unique(pLabels)
    accuracy = 0

    perms = np.array(list(permutations(np.unique(tLabels))))

    remappedLabels = tLabels
    remappedClasses = clusters
    for perm in perms:
        flippedLabels = np.zeros(len(tLabels), dtype=int)
        for l, label in enumerate(clusters):
            flippedLabels[pLabels == label] = perm[l]

        testAcc = evaluate(flippedLabels, tLabels)
        if testAcc > accuracy:
            accuracy = testAcc
            remappedLabels = flippedLabels
            remappedClasses = perm

    return accuracy, remappedLabels, remappedClasses

# map cluster labels using remapped classes
def mapLabels(pLabels, remappedClasses):
    clusters = np.unique(pLabels)
    mappedLabels = np.zeros(len(pLabels), dtype=int)
    for l, label in enumerate(clusters):
        mappedLabels[pLabels == label] = remappedClasses[l]
    return mappedLabels

# calculate the ratio of correctly predicted labels
def evaluate(predictions, truths):
    assert len(predictions) == len(truths)
    return np.sum(predictions == truths) / len(truths)
