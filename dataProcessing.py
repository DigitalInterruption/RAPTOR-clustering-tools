import os
import pandas as pd
import numpy as np

from scipy.stats import zscore

# collect batched feature data for individual samples into a complete list,
#   performing some preproncessing on it, recording details used later
def collateBatchData(dataset, featureMask,
        testSplitRatio = .2, testSplit = False, dropOutliers = False):
    dataDir = 'features/' + dataset + '/'
    families = os.listdir(dataDir)
    familyDataframes = []
    allSamples = []
    allClasses = []

    # loop through families, collecting sample features
    for f in families:
        samples = os.listdir(dataDir + f)
        sampleFrames = []
        for s in samples:
            sampleName = s.split('.')[0]
            allSamples.append(sampleName)
            sampleFrame = pd.read_csv(dataDir + f + '/' + s)
            allClasses.append(sampleFrame['family'].values[0])
            # add sample field for use later in this function
            sampleFrame['sample'] = sampleName
            sampleFrames.append(sampleFrame)

        # create a dataframe for each family
        df = pd.concat(sampleFrames, ignore_index=True, sort=False)
        
        # currently this solves the NaN issue by using the family column mean
        for col in df.columns:
            if df[col].dtypes != 'object':
                df[col].fillna(df[col].mean(), inplace=True)
        
        familyDataframes.append(df)

    # create a single dataframe
    df = pd.concat(familyDataframes, ignore_index=True, sort=False)

    # apply feature mask to dataset to include only the selected features
    df = dropFeatures(df, featureMask)

    # if flag passed remove data points containing outliers
    if dropOutliers: df = removeOutliers(df, True)

    # TODO: maybe add to drop outliers method as allSamples and allClasses are
    #   not used before this point, also generate those here in that case
    # loop through data, removing bad samples
    badSamples = []
    for i, s in enumerate(allSamples):
        sampleFrame = df.loc[df['sample'] == s]
        numPoints = sampleFrame.shape[0]
        # check that suffucient data points remain after outliers are removed,
        #   remove data points for samples with insufficient data points
        if numPoints <= 3 and dropOutliers: 
            badSamples.append([s, allClasses[i]])
            df = df.loc[df['sample'] != s]
    # remove bad samples from sample and class lists
    for b in badSamples:
        allSamples.remove(b[0])
        allClasses.remove(b[1])

    # create numerical mapping dict for classes
    classDict = {c : i for i, c in enumerate(np.unique(allClasses))}
    
    # calculate total number of data points in dataset
    numDataPoints = len(df)

    # get data point bounds for each sample
    sampleBounds = getBounds(df, allSamples)

    # perform final dataset operations and split sets if requested
    if testSplit:
        xData, xBounds, xSamples, yData, yBounds, ySamples =\
                splitData(df, sampleBounds, allSamples, allClasses,
                        testSplitRatio)

        # drop the sample field as it is no longer needed
        xData.drop(columns=['sample'], inplace=True)
        yData.drop(columns=['sample'], inplace=True)
        
        # execute final data preperation function and get total length
        xData, xLabels = produceDataSet(xData)
        yData, yLabels = produceDataSet(yData)

        # enumerate labels
        xLabels = enumerateLabels(xLabels, classDict)
        yLabels = enumerateLabels(yLabels, classDict)
        trueLabels = xLabels + yLabels

        # regenerate bounds for split datasets
        sampleBounds = xBounds
        offset = xBounds[-1][1] + 1
        for b in yBounds: sampleBounds.append([offset + b[0], offset + b[1]])
    # spoof test and train sets if no split requested
    else:
        xBounds, yBounds, xSamples, ySamples =\
                sampleBounds, sampleBounds, allSamples, allSamples

        # drop the sample field as it is no longer needed
        df.drop(columns=['sample'], inplace=True)
        
        # execute final data preperation function and get total length
        df, labels = produceDataSet(df)

        # enumerate labels
        labels = enumerateLabels(labels, classDict)

        xData, yData, xLabels, yLabels = df, df, labels, labels
        trueLabels = labels

    numSamples = len(sampleBounds)

    return xData, xLabels, xBounds, xSamples,\
            yData, yLabels, yBounds, ySamples,\
            numSamples, sampleBounds, numDataPoints, trueLabels, classDict

# collect the family feature sets into a dataset for clustering
def collateFamilyData(dataDir, dropOutliers = False):
    df = pd.read_csv(dataDir + 'results.csv')
    if dropOutliers: df = removeOutliers(df)
    return produceDataSet(df)

# drop features from data not specified in the feature mask
def dropFeatures(data, featureMask):
    droppedFeatures = [
            'node-weight',
            'in-degree',
            'out-degree',
            'degree',
            'closeness-wtd',
            'in-closeness-wtd',
            'out-closeness-wtd',
            'closeness-unwtd',
            'in-closeness-unwtd',
            'out-closeness-unwtd',
            'betweenness-wtd',
            'betweenness-unwtd',
            'first-order-influence-wtd',
            'first-order-influence-unwtd',
            'clustering-coefficient'
            ]
    
    #print('features = ', featureMask)
    
    [droppedFeatures.remove(f) for f in featureMask]
    data.drop(columns=droppedFeatures, inplace=True)

    return data

# extract labels and remove the corresponding fields
def produceDataSet(df):
    # remove node column
    df.drop(columns=['node'], inplace=True)
    # currently this solves the NaN issue by using the column mean
    for col in df.columns:
        if df[col].dtypes != 'object':
            df[col].fillna(df[col].mean(), inplace=True)
    # collect class information to list and remove from dataset
    trueLabels = [l for l in df['family']]
    data = df.drop(columns=['family'])
    return data, trueLabels

# remove outliers from the dataset by z-score threshold
def removeOutliers(data, samples = False, thresh = 3):
    if samples:
        return data[(np.abs(zscore(data[data.columns.drop(['family', 'node', 'sample'])])) < thresh).all(axis=1)]
    else:
        return data[(np.abs(zscore(data[data.columns.drop(['family', 'node'])])) < thresh).all(axis=1)]

# get index bounds for data points pertaining to each sample
def getBounds(data, samples):
    bounds, start = [], 0
    for s in samples:
        num = data.loc[data['sample'] == s].shape[0]
        end = start + num
        bounds.append([start, end])
        start = end
    return bounds

# enumerate class labels
def enumerateLabels(labels, classDict): return [classDict[c] for c in labels]

# split data into test (y) and train (x) sets by the provided ratio, preserving
#   class distributions, warning where there are few test samples
def splitData(data, bounds, samples, classes, ratio):

    # create randomised index list
    numSamples = len(samples)
    rng = np.random.default_rng()
    indexes = np.arange(numSamples)
    rng.shuffle(indexes)
    indexes = list(indexes)

    # calculate target test-set sizes for each class, create counter dict
    classNames, classCounts = np.unique(np.array(classes), return_counts=True)
    targets = {i[0] : int(i[1] * ratio) for i in zip(classNames, classCounts)}
    quantities = {n : 0 for n in classNames}
    
    # iterate through the index list evaluating if sample should be included in
    #   testing set, if not add to training set
    xBounds, yBounds, xSamples, ySamples = [], [], [], []
    for i in indexes:
        if targets[classes[i]] > quantities[classes[i]]:
            yBounds.append(bounds[i])
            ySamples.append(samples[i])
            quantities[classes[i]] += 1
        else:
            xBounds.append(bounds[i])
            xSamples.append(samples[i])

    # split dataset into test (y) and train (x) sets
    xDataList, yDataList = [], []
    for b in xBounds: xDataList.append(data.iloc[b[0]:b[1]])
    for b in yBounds: yDataList.append(data.iloc[b[0]:b[1]])
    xData = pd.concat(xDataList, ignore_index=True, sort=False)
    yData = pd.concat(yDataList, ignore_index=True, sort=False)

    # regenerate bounds for split datasets
    xBounds = getBounds(xData, xSamples)
    yBounds = getBounds(yData, ySamples)

    return xData, xBounds, xSamples,\
            yData, yBounds, ySamples

# normalise data by selected method
def normaliseData(X, Y, method):
    for col in X.columns:
        if method == 'min-max':
            X[col] = (X[col] - X[col].min()) / (X[col].max() - X[col].min())
            Y[col] = (Y[col] - Y[col].min()) / (Y[col].max() - Y[col].min())
        if method == 'max-abs':
            X[col] = X[col] / X[col].abs().max()
            Y[col] = Y[col] / Y[col].abs().max()
        if method == 'zscaled':
            X[col] = (X[col] - X[col].mean()) / X[col].std()
            Y[col] = (Y[col] - Y[col].mean()) / Y[col].std()
    
    return X, Y
