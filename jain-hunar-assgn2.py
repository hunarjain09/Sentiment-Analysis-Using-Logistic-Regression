import random as rd
import matplotlib.pyplot as plt
import numpy as np


##################################################
def getFeatureVectorsFromCSV():
    with open("jain-hunar-assgn2-part1.csv",'r') as file_name:
        featureVectors=  np.genfromtxt('exp2.csv', delimiter=',', dtype=None,encoding=None)
    return featureVectors

def shuffleFeatureVectors(featureVectors):
    # rd.seed(30)
    featureVectors = list(map(list,featureVectors))
    rd.shuffle(featureVectors)
    return featureVectors

##################################################

def splitToTrainAndDev(featureVectors):
    limit = int(0.80*len(featureVectors))
    train_set = featureVectors[:limit]
    dev_set = featureVectors[limit:]
    # print(len(test_set),len(dev_set))
    return train_set,dev_set

##################################################    
def sigmoid(X,w):
    z = np.dot(w,X)
    return 1/(1+np.exp(-(z)))

def crossEntropyLoss(X, y, w):
    y1 = sigmoid(X, w)
    # print(y,y1)
    return -(y*np.log(y1) + (1-y)*np.log(1-y1))

def gradient_descent(X, w, learningRate, epochs):
    for i in range(0, epochs):
        lossTemp = []
        accTemp = []
        for j in range(len(X)):
            # print(X[j])
            y = X[j][-1]
            actualAnswer = 'POS' if y == 1 else 'NEG'
            featureVector = X[j][1:-1]+[1]
            # print(featureVector)
            h = sigmoid(featureVector, w)
            if h < 0.5:
                accTemp.append((actualAnswer,'NEG'))
            else:
                accTemp.append((actualAnswer,'POS'))
            for k in range(0, len(featureVector)):
                w[k] -= (learningRate) * ((h-y)*featureVector[k])
            
            lossTemp.append(crossEntropyLoss(featureVector,y,w))

        correctList = list(map(lambda x: 1,filter(lambda x: x[0] == x[1], accTemp)))
        accuracies.append(sum(correctList)/len(accTemp))
        losses.append(sum(lossTemp)/len(X))
    return w

def predict_Train(X,w):
    with open('jain-hunar-assgn2-out-2.txt', 'w') as file:
        for j in range(len(X)):
            toWrite = [X[j][0]]
            y = 'POS' if X[j][-1] == 1 else 'NEG'
            featureVector = X[j][1:-1]+[1]
            # print(featureVector)
            h = sigmoid(featureVector, w)
            if h < 0.5:
                sanityTestList.append((y,'NEG',toWrite))
                toWrite.append('NEG')
            else:
                sanityTestList.append((y,'POS',toWrite))
                toWrite.append('POS')

            file.write('\t'.join(toWrite) + '\n')
    return

def predict_Test(X,w):
    with open('jain-hunar-assgn2-out-1.txt', 'w') as file:
        for j in range(len(X)):
            toWrite = [X[j][0]]
            featureVector = X[j][1:]+[1]
            h = sigmoid(featureVector, w)
            if h < 0.5:
                toWrite.append('NEG')
            else:
                toWrite.append('POS')

            file.write('\t'.join(toWrite) + '\n')
    return

####################################################

def plotLossesAndAccuracies(losses, accuracies):
    # print(losses)
    myLosses = plt.figure("Losses")
    plt.plot(losses)

    # print(accuracies)
    myAccuracies = plt.figure("Accuracies")
    plt.plot(accuracies)

    plt.show()
    
    return

####################################################

def filterFinalPredictions(sanityTestList):
    filtered = list(filter(lambda x: x[0]!= x[1], sanityTestList))
    return len(sanityTestList),len(filtered),filtered

##################### FEATURE_VECTORS ########################

featureVectors = getFeatureVectorsFromCSV()
featureVectors = shuffleFeatureVectors(featureVectors)
train_set, dev_set = splitToTrainAndDev(featureVectors)

##################### INITIALISATION ########################

w = [0,0,0,0,0,0,0]
losses = []
accuracies = []
sanityTestList = []

##################### LEARNING ########################
# 

# Todo: Remove train_set and replace it with entire feature_vectors.
gradient_descent(train_set,w,0.01,25)

##################### UTILITIES ########################

# PRINTING WEIGHTS AND DEV SET

# TO PRINT WEIGHTS
# print(w)


# TO PRINT DEV_SET
# print(dev_set)

##################### PREDICTION ON DEV_SET AND PLOTTING ########################

# predict_Train(dev_set,w)
# print(filterFinalPredictions(sanityTestList))
# plotLossesAndAccuracies(losses, accuracies)

##################### PREDICTION ON TEST ########################

testFeatureVectors = getFeatureVectorsFromCSV()

predict_Test(testFeatureVectors,w)
