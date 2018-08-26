# Neural Network Classifier

A three-layer neural network classifier implemented in Python using only the NumPy library.
By utilizing the parameters in the original file, it is able to get an 82.83% testing accuracy on the MNIST dataset. In the future improvements section below, I specify how I plan to improve the accuracy of this network.


## Neural Network Design

This neural network has three layers, an input layer, a hidden layer and an output layer. The number of nodes in each layer is completely customizable with the parameters provided at the bottom of the Python file. This implementation of the network considers bias.

The network first obtains the data and separates the features from the target variable. It then scales the features with mean normalization. Any features that do not contain useful data (where all values are identical for all training examples) are trimmed. The training examples are then passed to the `train` function of an instance of the class `NNMulticlass`.

The target variable `y` is passed through a one-hot encoder. The network then uses a vectorized implementation of forward propagation and backpropagation to update the weights. The cost is computed with the logistic cross-entropy error function. Note that the gradients and the cost are regularized. The cost is minimized with batch gradient descent.

After training with the training data, the test data is passed through forward propagation to compute the accuracy. The predict function can be used to predict an outcome `y` given an input `X`. 


## Training Data Format

The training data should be formatted as follows. Each row is a new training example with comma-delimited data (I would recommend using a CSV file). The first `n - 1` numbers (where `'n` is the number of columns) represent the features (X<sup>(i)</sup>). The last number represents the classification for that training example (y<sup>(i)</sup>).


## Future Improvements

I created this network after completing the Machine Learning course offered through Coursera and Stanford Online that is taught by Andrew Ng. My objective was to gain further insight in the process of creating and optimizing a neural network. I have therefore identified key areas where this network can be improved.
* Use another activation function instead of the sigmoid function for the intermediate layers of the network (perhaps tanh, ReLU or softmax).
* Implement a more sophisticated regularization for the cost and gradient (L1 and L2 regularization).
* Add mini-batch and stochastic gradient descent options.
* Generate an option to decrease the learning rate as a function of epochs for faster initial learning.
* Employ another more advanced optimization function (conjugate gradient, BFGS, L-BFGS).
* Increase and parametrize the number of layers in the network.
* Customize the formats for the training and testing data.
* Add an option to print the weights into a file and to load weights from a file for forward propagation.


