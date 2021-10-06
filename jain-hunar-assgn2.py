import random as rd
import matplotlib.pyplot as plt
import numpy as np


##################################################
class Utilities:
    def __init__(self) -> None:
        return

    def getFeatureVectorsFromCSV(self,fileName):
        with open(fileName,'r') as file_name:
            featureVectors=  np.genfromtxt('exp2.csv', delimiter=',', dtype=None,encoding=None)
        return featureVectors

    def shuffleFeatureVectors(self,featureVectors):
        # rd.seed(30)
        featureVectors = list(map(list,featureVectors))
        rd.shuffle(featureVectors)
        return featureVectors

    def splitToTrainAndDev(self,featureVectors):
        limit = int(0.80*len(featureVectors))
        train_set = featureVectors[:limit]
        dev_set = featureVectors[limit:]
        # print(len(test_set),len(dev_set))
        return train_set,dev_set

    def plotLossesAndAccuracies(self,losses, accuracies):
        # print(losses)
        myLosses = plt.figure("Losses")
        plt.plot(losses)

        # print(accuracies)
        myAccuracies = plt.figure("Accuracies")
        plt.plot(accuracies)

        plt.show()
        
        return
    
    def filterFinalPredictions(self,sanityTestList):
        filtered = list(filter(lambda x: x[0]!= x[1], sanityTestList))
        return len(sanityTestList),len(filtered),filtered

##################################################

class Logistic_Regression:
    def __init__(self) -> None:
        self.losses = []
        self.accuracies = []
        self.sanityTestList = []
        return
        
    def sigmoid(self,X,w):
        z = np.dot(w,X)
        return 1/(1+np.exp(-(z)))

    def crossEntropyLoss(self,X, y, w):
        y1 = self.sigmoid(X, w)
        # print(y,y1)
        return -(y*np.log(y1) + (1-y)*np.log(1-y1))

    def gradient_descent(self,X, w, learningRate, epochs):
        for i in range(0, epochs):
            lossTemp = []
            accTemp = []
            for j in range(len(X)):
                # print(X[j])
                y = X[j][-1]
                actualAnswer = 'POS' if y == 1 else 'NEG'
                featureVector = X[j][1:-1]+[1]
                # print(featureVector)
                h = self.sigmoid(featureVector, w)
                if h < 0.5:
                    accTemp.append((actualAnswer,'NEG'))
                else:
                    accTemp.append((actualAnswer,'POS'))
                for k in range(0, len(featureVector)):
                    w[k] -= (learningRate) * ((h-y)*featureVector[k])
                
                lossTemp.append(self.crossEntropyLoss(featureVector,y,w))

            correctList = list(map(lambda x: 1,filter(lambda x: x[0] == x[1], accTemp)))
            self.accuracies.append(sum(correctList)/len(accTemp))
            self.losses.append(sum(lossTemp)/len(X))
        return w

    def predict_Train(self,X,w):
        with open('jain-hunar-assgn2-out-2.txt', 'w') as file:
            for j in range(len(X)):
                toWrite = [X[j][0]]
                y = 'POS' if X[j][-1] == 1 else 'NEG'
                featureVector = X[j][1:-1]+[1]
                # print(featureVector)
                h = self.sigmoid(featureVector, w)
                if h < 0.5:
                    self.sanityTestList.append((y,'NEG',toWrite))
                    toWrite.append('NEG')
                else:
                    self.sanityTestList.append((y,'POS',toWrite))
                    toWrite.append('POS')

                file.write('\t'.join(toWrite) + '\n')
        return

    def predict_Test(self,X,w):
        with open('jain-hunar-assgn2-out-1.txt', 'w') as file:
            for j in range(len(X)):
                toWrite = [X[j][0]]
                featureVector = X[j][1:]+[1]
                h = self.sigmoid(featureVector, w)
                if h < 0.5:
                    toWrite.append('NEG')
                else:
                    toWrite.append('POS')

                file.write('\t'.join(toWrite) + '\n')
        return

##################### FEATURE_VECTORS ########################

myUtilities = Utilities()
featureVectors = myUtilities.getFeatureVectorsFromCSV("jain-hunar-assgn2-part1.csv")
featureVectors = myUtilities.shuffleFeatureVectors(featureVectors)
train_set, dev_set = myUtilities.splitToTrainAndDev(featureVectors)

##################### INITIALISATION ########################

w = [0,0,0,0,0,0,0]

##################### LEARNING ########################

# Todo: Remove train_set and replace it with entire feature_vectors.
myLogisticRegression = Logistic_Regression()
w = myLogisticRegression.gradient_descent(train_set,w,0.01,25)

##################### UTILITIES ########################

# PRINTING WEIGHTS AND DEV SET

# TO PRINT WEIGHTS
# print(w)


# TO PRINT DEV_SET
# print(dev_set)

##################### PREDICTION ON DEV_SET AND PLOTTING ########################

myLogisticRegression.predict_Train(dev_set,w)
print(myUtilities.filterFinalPredictions(myLogisticRegression.sanityTestList))
myUtilities.plotLossesAndAccuracies(myLogisticRegression.losses, myLogisticRegression.accuracies)

##################### PREDICTION ON TEST ########################

# testFeatureVectors = myUtilities.getFeatureVectorsFromCSV()

# myLogisticRegression.predict_Test(testFeatureVectors,w)
