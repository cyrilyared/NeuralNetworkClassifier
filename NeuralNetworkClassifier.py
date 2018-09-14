import numpy as np

class NNMulticlass:
    def __init__(self, nodes, learningParams, randomSeed, weights):
        """Initializes instance of NNMultiClass.
        
        Args:
            nodes: Tuple containing size of (input node, hidden node, output node).
            learningParams: Tuple containing (lambda, number of epochs, learning rate).
            randomSeed: Either int containing random seed or None.
            weights: Tuple of NumPy arrays containing (weight1, weight2) or None.
        """
        self.inputNode, self.hiddenNode, self.outputNode = nodes
        self.lambd, self.epochs, self.learningRate = learningParams
        if randomSeed:
            np.random.seed(randomSeed)
        if weights:
            self.weight1, self.weight2 = weights
        else:
            self.weight1, self.weight2 = self._generateWeights()
    
    def _generateWeights(self):
        """Returns randomly generated weights.
        
        Returns:
            weight1: NumPy array containing randomly generated floats.
            weight2: Numpy array containing randomly generated floats.
        """
        weight1 = np.random.uniform(-1.0, 1.0, size=(self.hiddenNode, self.inputNode + 1))
        weight2 = np.random.uniform(-1.0, 1.0, size=(self.outputNode, self.hiddenNode + 1))
        return weight1, weight2
            
    def _addBiasCol(self, X):
        """Returns a column of ones appended to the left of X.
        
        Args:
            X: NumPy array containing features.
            
        Returns:
            NumPy array with column of ones appended to left of X.
        """
        return np.column_stack((np.ones((X.shape[0], 1), dtype=int), X))

    def _oneHotEncoder(self, label):
        """Returns one-hot encoded array with data that corresponds to label.
        
        Args:
            label: NumPy array with labels corresponding to training or testing examples.
            
        Returns:
            maskMatrix: NumPy array with one-hot encoded data.
        """
        maskMatrix = np.zeros((label.size, self.outputNode), dtype=np.int)
        maskMatrix[np.arange(label.size), label] = 1
        return maskMatrix

    def _sigmoid(self, X):
        """Returns sigmoid function of X.
            
        Args:
            X: Float or NumPy array.
            
        Returns:
            Sigmoid of X.
        """
        return 1/(1+np.exp(-X))

    def _derivativeSigmoid(self, X):
        """Returns derivative of sigmoid function of X.
            
        Args:
            X: Float or NumPy array.
            
        Returns:
            Derivative of sigmoid of X.
        """
        return self._sigmoid(X)*(1-self._sigmoid(X))

    def _logisiticCrossEntropyRegularized(self, activation, y):
        """Returns regularized cross-entropy error.
            
        Args:
            activation: NumPy array containing hypothesis.
            y: One-hot encoded target variable.
            
        Returns:
            Cross-entropy error.
        """
        m = y.shape[0]
        J = (-1/float(m))*np.sum(y*(np.log(activation))+(1-y)*np.log(1-activation))
        regularizationCost = (np.sum(np.square(self.weight1[:,1:])) + np.sum(np.square(self.weight2[:,1:])))*(self.lambd/float(2*m))
        return J + regularizationCost

    def _forwardPropagation(self, X):
        """Compute forward propagation.
        
        Args:
            X: NumPy array containing features.
            
        Returns:
            activation1: NumPy array with activation of layer 1.
            z2: NumPy array with z of layer 2.
            activation2: NumPy array with activation of layer 2.
            z3: NumPy array with z of layer 3.
            activation3: NumPy array with activation of layer 3 (hypothesis).
        """
        activation1 = self._addBiasCol(X)
        z2 = activation1.dot(np.transpose(self.weight1))
        activation2 = self._addBiasCol(self._sigmoid(z2))
        z3 = activation2.dot(np.transpose(self.weight2))
        activation3 = self._sigmoid(z3)
        return activation1, z2, activation2, z3, activation3

    def _backpropagation(self, activation1, z2, activation2, activation3, y):
        """Compute backpropagation.
        
        Args:
            activation1: Activation of layer 1.
            z2: Z of layer 2.
            activation2: Activation of layer 2.
            activation3: Activation of layer 3 (hypothesis).
            y: One-hot encoded target variable.
        
        Returns:
            delta1: Error for weight1.
            delta2: Error for weight2.
        """
        sigma3 = activation3 - y;
        sigma2 = sigma3.dot(self.weight2[:,1:])*self._derivativeSigmoid(z2)
        delta1 = np.transpose(sigma2).dot(activation1)
        delta2 = np.transpose(sigma3).dot(activation2)
        return delta1, delta2

    def _trainingStep(self, X, y):
        """Computes one epoch of training and cost.
        
        Args:
            X: NumPy array containing features.
            y: One-hot encoded target variable.
        
        Returns:
            J: Regularized cross-entropy error.
            delta1: Regularized gradient for weight1.
            delta2: Regularized gradient for weight2.
        """
        activation1, z2, activation2, z3, activation3 = self._forwardPropagation(X)
        delta1, delta2 = self._backpropagation(activation1, z2, activation2, activation3, y)
        m = X.shape[0]
        delta1 = delta1/float(m) + np.column_stack((np.zeros((delta1.shape[0], 1), dtype=int), self.weight1[:,1:]))*self.lambd/float(m)
        delta2 = delta2/float(m) + np.column_stack((np.zeros((delta2.shape[0], 1), dtype=int), self.weight2[:,1:]))*self.lambd/float(m)
        J = self._logisiticCrossEntropyRegularized(activation3, y)
        return J, delta1, delta2

    def train(self, X, y):
        """Trains neural network.
            
        Args:
            X: NumPy array containing features.
            y: NumPy array with labels corresponding to training or testing examples.
        """
        self.cost = []
        Xtrain, ytrain = X.copy(), self._oneHotEncoder(y.copy())
        for i in range(self.epochs):
            cost, gradient1, gradient2 = self._trainingStep(Xtrain, ytrain)
            self.weight1 = self.weight1 - gradient1*self.learningRate
            self.weight2 = self.weight2 - gradient2*self.learningRate
            self.cost.append(cost)
        return self

    def predict(self, X):
        """Computes hypothesis based on X.
        
        Args:
            X: NumPy array containing features.
            
        Returns:
            NumPy array with predicted labels based on input.
        """
        Xtest = X.copy()
        activation1, z2, activation2, z3, activation3 = self._forwardPropagation(Xtest)
        return np.argmax(activation3, axis = 1)

    def softmaxProbability(self, X):
        """Computes the softmax function of the hypothesis.
        
        Args:
            X: NumPy array containing features.
        
        Returns:
            NumPy array with the softmax of the hypothesis.
        """
        Xtest = X.copy()
        activation1, z2, activation2, z3, activation3 = self._forwardPropagation(Xtest)
        return np.exp(np.transpose(activation3))/np.sum(np.exp(np.transpose(activation3)))

    def accuracy(self, X, y):
        """Returns the accuracy of the network on the training or testing examples.
        
        Args:
            X: NumPy array containing features.
            y: NumPy array with labels corresponding to training or testing examples.
            
        Returns:
            Float of the accuracy of the network.
        """
        hypothesis = self.predict(X)
        return np.sum(y == hypothesis, axis=0)/float(X.shape[0])

    def saveWeights(self):
        """Saves weight1 and weight2 into a CSV file."""
        np.savetxt(raw_input("Enter the filename for the document that will contain weight1: "), self.weight1, delimiter = ',')
        np.savetxt(raw_input("Enter the filename for the document that will contain weight2: "), self.weight2, delimiter = ',')


def getData(filetype, datatype):
    """Returns NumPy array containing data from CSV file.
        
    Args:
        filetype: String describing the file to be opened.
        datatype: Datatype of NumPy array.
        
    Returns:
        NumPy array with data from CSV file in format datatype.
    """
    filename = raw_input("What is the filename of the file containing the " + filetype +  " data: ")
    return np.genfromtxt(filename, delimiter=',').astype(datatype)

def formatFeatureTarget(array):
    """Returns two NumPy arrays containing features and target variable.
   
    Args:
        array: NumPy array with features and target variable.
    
    Returns:
        X: NumPy array with features.
        y: NumPy array with target variable.
    """
    X, y = trainingData[:, :-1], trainingData[:, -1].astype(int)
    return X, y

def loadWeights():
    """Returns saved weight1 and weight2 from file.
        
    Returns:
        weight1: NumPy array containing weight1.
        weight2: Numpy array containing weight2.
    """
    weight1 = getData("weight1", float)
    weight2 = getData("weight2", float)
    return weight1, weight2

def saveMinMaxMeanTrim(min, max, mean, trimIndex):
    """Saves min, max, mean and trim index data to file.
        
    Args:
        min: NumPy array of minimum of each column of features.
        max: NumPy array of maximum of each column of features.
        mean: NumPy array of mean of each column of features.
        trimIndex: List containing row or column numbers that were trimmed.
    """
    trimIndex = np.asarray(trimIndex)
    saveArray = np.vstack((min, max, mean, np.hstack((trimIndex, np.zeros(mean.size-trimIndex.size))), np.hstack((trimIndex.size, np.zeros(mean.size-1)))))
    np.savetxt(raw_input("Enter the filename for the document that will contain the min, max, mean and trim index: "), saveArray, delimiter = ',')

def loadMinMaxMeanTrim():
    """Returns min, max, mean and trim index data from file.
    
    Returns:
        min: NumPy array of minimum of each column of features.
        max: NumPy array of maximum of each column of features.
        mean: NumPy array of mean of each column of features.
        trimIndex: List containing row or column numbers that were trimmed.
    """
    array = getData("min, max, mean and trim index", float)
    min, max, mean = array[0,:], array[1,:], array[2,:]
    trimIndexSize = array[4,0]
    trimIndex = array[3,0:trimIndexSize.astype(int)]
    return min, max, mean, trimIndex.tolist()

def checkDimensions(array):
    """Returns array with two dimensions for matrix operations.
    
    Args:
        array: Input NumPy array.
    
    Returns:
        array: NumPy array with two dimensions.
    """
    if array.ndim == 1:
        array = array[np.newaxis]
    return array

def calculateMinMaxMean(X):
    """Returns min, max and mean of each column of X.
        
    Args:
        X: NumPy array containing features.
        
    Returns:
        min: NumPy array with minimum of each column of X.
        max: NumPy array with maximum of each column of X.
        mean: NumPy array with mean of each column of X.
    """
    min = np.amin(X, axis=0)
    max = np.amax(X, axis=0)
    mean = np.mean(X, axis=0)
    return min, max, mean

def trimFeatures(min, max, mean, X):
    """Determines which features to trim (data is constant for feature) and returns trimmed version of min, max, mean, X.
        
    Args:
        min: NumPy array with minimum of each column of X.
        max: NumPy array with maximum of each column of X.
        mean: NumPy array with mean of each column of X.
        X: NumPy array containing features.
        
    Returns:
        min: NumPy array with trimmed minimum of each column of X.
        max: NumPy array with trimmed maximum of each column of X.
        mean: NumPy array with trimmed mean of each column of X.
        X: Trimmed NumPy array containing features.
        trimIndex: List containing row or column numbers that were trimmed.
    """
    trimIndex = []
    for i in range(min.shape[0]):
        if (max[i]-min[i]) == 0:
            trimIndex.append(i)
    X = trim(X, trimIndex, axis=1)
    min = trim(min, trimIndex)
    max = trim(max, trimIndex)
    mean = trim(mean, trimIndex)
    return min, max, mean, X, trimIndex

def trim(array, trimIndex, axis=None):
    """Returns array after trimming rows or columns.
        
    Args:
        array: NumPy array to be trimmed.
        trimIndex: List containing row or column numbers to be trimmed.
        axis: Axis to be trimmed (default None).
        
    Returns:
        NumPy array with trimmed rows or columns.
    """
    return np.delete(array, trimIndex, axis)

def normalizeData(min, max, mean, X):
    """Returns the features X after mean normalization.
        
    Args:
        min: NumPy array with minimum of each column of X.
        max: NumPy array with maximum of each column of X.
        mean: NumPy array with mean of each column of X.
        X: NumPy array containing features.
        
    Returns:
        X: NumPy array containing mean-normalized features.
    """
    X = X.astype(float)
    for i in range(X.shape[0]):
        X[i] = (X[i] - mean)/((max-min).astype(float))
    return X

if __name__ == "__main__":
    trainingData = getData("training", float)
    trainingData = checkDimensions(trainingData)
    Xtrain, ytrain = formatFeatureTarget(trainingData)

    min, max, mean = calculateMinMaxMean(Xtrain) # Replace with loadMinMeanMaxTrim() to use custom data from file.
    
    # If loadMinMeanMaxTrim() called, replace line below with Xtrain = trim(Xtest, trimIndex, axis=1)
    min, max, mean, Xtrain, trimIndex = trimFeatures(min, max, mean, Xtrain)
    
    Xtrain = normalizeData(min, max, mean, Xtrain)
    
    # Parameters for the neural network.
    hiddenNode = 250
    outputNode = 10
    lambd = 0.3
    epochs = 500
    learningRate = 0.08
    randomSeed = None
    inputNode = Xtrain.shape[1]
    weights = None  # Replace with loadWeights() to use custom weights from file.
    
    NN = NNMulticlass((inputNode, hiddenNode, outputNode), (lambd, epochs, learningRate), randomSeed, weights)

    NN.train(Xtrain, ytrain)
    print("Training Accuracy: %.2f%%" % (NN.accuracy(Xtrain, ytrain)*100))
    
    testingData = getData("testing", float)
    testingData = checkDimensions(testingData)
    Xtest, ytest = formatFeatureTarget(trainingData)
    Xtest = normalizeData(min, max, mean, trim(Xtest, trimIndex, axis=1))
    print("Testing Accuracy: %.2f%%" % (NN.accuracy(Xtest, ytest)*100))
