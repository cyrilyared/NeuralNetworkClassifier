import numpy as np

class NNMulticlass:
    def __init__(self, nodes, learningParams, randomSeed):
        self.inputNode, self.hiddenNode, self.outputNode = nodes
        self.lambd, self.epochs, self.learningRate = learningParams
        if randomSeed:
            np.random.seed(randomSeed)
        self.weight1, self.weight2 = self._generateWeights()
    
    def _generateWeights(self):
        weight1 = np.random.uniform(-1.0, 1.0, size=(self.hiddenNode, self.inputNode + 1))
        weight2 = np.random.uniform(-1.0, 1.0, size=(self.outputNode, self.hiddenNode + 1))
        return weight1, weight2
            
    def _addBiasCol(self, X):
        return np.column_stack((np.ones((X.shape[0], 1), dtype=int), X))

    def _oneHotEncoder(self, label):
        maskMatrix = np.zeros((label.size, self.outputNode), dtype=np.int)
        maskMatrix[np.arange(label.size), label] = 1
        return maskMatrix

    def _sigmoid(self, X):
        return 1/(1+np.exp(-X))

    def _derivativeSigmoid(self, X):
        return self._sigmoid(X)*(1-self._sigmoid(X))

    def _logisiticCrossEntropyRegularized(self, activation, y):
        m = y.shape[0]
        J = (-1/float(m))*np.sum(y*(np.log(activation))+(1-y)*np.log(1-activation))
        regularizationCost = (np.sum(np.square(self.weight1[:,1:])) + np.sum(np.square(self.weight2[:,1:])))*(self.lambd/float(2*m))
        return J + regularizationCost

    def _forwardPropogation(self, X):
        activation1 = self._addBiasCol(X)
        z2 = activation1.dot(np.transpose(self.weight1))
        activation2 = self._addBiasCol(self._sigmoid(z2))
        z3 = activation2.dot(np.transpose(self.weight2))
        activation3 = self._sigmoid(z3)
        return activation1, z2, activation2, z3, activation3

    def _backpropogation(self, activation1, z2, activation2, activation3, y):
        sigma3 = activation3 - y;
        sigma2 = sigma3.dot(self.weight2[:,1:])*self._derivativeSigmoid(z2)
        delta1 = np.transpose(sigma2).dot(activation1)
        delta2 = np.transpose(sigma3).dot(activation2)
        return delta1, delta2

    def _trainingStep(self, X, y):
        activation1, z2, activation2, z3, activation3 = self._forwardPropogation(X)
        delta1, delta2 = self._backpropogation(activation1, z2, activation2, activation3, y)
        m = X.shape[0]
        delta1 = delta1/float(m) + np.column_stack((np.zeros((delta1.shape[0], 1), dtype=int), self.weight1[:,1:]))*self.lambd/float(m)
        delta2 = delta2/float(m) + np.column_stack((np.zeros((delta2.shape[0], 1), dtype=int), self.weight2[:,1:]))*self.lambd/float(m)
        J = self._logisiticCrossEntropyRegularized(activation3, y)
        return J, delta1, delta2

    def train(self, X, y):
        self.cost = []
        Xtrain, ytrain = X.copy(), self._oneHotEncoder(y.copy())
        for i in range(self.epochs):
            cost, gradient1, gradient2 = self._trainingStep(Xtrain, ytrain)
            self.weight1 = self.weight1 - gradient1*self.learningRate
            self.weight2 = self.weight2 - gradient2*self.learningRate
            self.cost.append(cost)
        return self

    def predict(self, X):
        Xtest = X.copy()
        activation1, z2, activation2, z3, activation3 = self._forwardPropogation(Xtest)
        return np.argmax(activation3, axis = 1)

    def softmaxProbability(self, X):
        Xtest = X.copy()
        activation1, z2, activation2, z3, activation3 = self._forwardPropogation(Xtest)
        return np.exp(np.transpose(activation3))/np.sum(np.exp(np.transpose(activation3)))

    def accuracy(self, X, y):
        hypothesis = self.predict(X)
        return np.sum(y == hypothesis, axis=0)/float(X.shape[0])


def getData(type):
    filename = raw_input("What is the filename of the file containing the " + type +  " data: ")
    return np.genfromtxt(filename, delimiter=',').astype(int)

def checkDimensions(array):
    if array.ndim == 1:
        array = array[np.newaxis]
    return array

def getMinMaxMean(X):
    min = np.amin(X, axis=0)
    max = np.amax(X, axis=0)
    mean = np.mean(X, axis=0)
    return min, max, mean

def trimFeatures(min, max, mean, X):
    trimIndex = []
    for i in range(min.shape[0]):
        if (max[i]-min[i]) == 0:
            trimIndex.append(i)
    X = np.delete(X, trimIndex, axis=1)
    min = np.delete(min, trimIndex)
    max = np.delete(max, trimIndex)
    mean = np.delete(mean, trimIndex)
    return min, max, mean, X, trimIndex

def normalizeData(min, max, mean, X):
    X = X.astype(float)
    for i in range(X.shape[0]):
        X[i] = (X[i] - mean)/((max-min).astype(float))
    return X

if __name__ == "__main__":
    trainingData = getData("training")
    trainingData = checkDimensions(trainingData)
    Xtrain, ytrain = trainingData[:, :-1], trainingData[:, -1]

    min, max, mean = getMinMaxMean(Xtrain)
    min, max, mean, Xtrain, trimIndex = trimFeatures(min, max, mean, Xtrain)
    Xtrain = normalizeData(min, max, mean, Xtrain)
    
    hiddenNode = 250
    outputNode = 10
    lambd = 0.3
    epochs = 500
    learningRate = 0.08
    randomSeed = None
    inputNode = Xtrain.shape[1]
    
    NN = NNMulticlass((inputNode, hiddenNode, outputNode), (lambd, epochs, learningRate), randomSeed)

    NN.train(Xtrain, ytrain)
    print("Training Accuracy: %.2f%%" % (NN.accuracy(Xtrain, ytrain)*100))
    
    testingData = getData("testing")
    testingData = checkDimensions(testingData)
    Xtest, ytest = testingData[:, :-1], testingData[:, -1]
    Xtest = normalizeData(min, max, mean, np.delete(Xtest, trimIndex, axis=1))
    print("Testing Accuracy: %.2f%%" % (NN.accuracy(Xtest, ytest)*100))
