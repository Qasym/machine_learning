Jimport numpy as np
import matplotlib.pyplot as plt

'''
input layer consists of 2 neurons
hidden layer consists of 150 neurons
so the weights1 variable that is between input layer
and hidden layer should be 2x150
since my design implies one hidden layer
and we have only binary output (either 0 or 1)
weights2 variable should be 150x1
weights2 is a matrix between the output layer and the hidden layer
'''

class NeuralNetwork:
    def __init__(self, hiddenLayerNodes, learningRate, epochs=None):
        # Assign parameters
        self.learningRate = learningRate
        self.epochs = epochs

        # Initialize the weights
        self.weights_1 = np.random.rand(2, hiddenLayerNodes) # from input to hidden
        self.weights_2 = np.random.rand(hiddenLayerNodes, 1) # from hidden to output

        # Initialize the bias
        self.bias_1 = np.random.rand(1, hiddenLayerNodes) # from input to hidden
        self.bias_2 = np.random.rand(1, 1) # from hidden to output

        # Loss over time tracking
        self.train_loss = []
        self.test_loss = []

    def load_data(self):
        # Load data
        train = np.loadtxt('Trn.csv', delimiter=',')
        test = np.loadtxt('Tst.csv', delimiter=',')
        grid = np.loadtxt('Grid.csv', delimiter=',')

        # Obtain training data
        trainX = train[:, :2]
        trainY = train[:, 2]
        trainY = trainY.reshape(len(trainY), 1)

        # Set epochs
        if self.epochs == None:
            self.epochs = len(trainX)

        # Obtain test data
        testX = test[:, :2]
        testY = test[:, 2]
        testY = testY.reshape(len(testY), 1)

        # We don't need to separate grid data, since it is only X without Y

        return trainX, trainY, testX, testY, grid

    def save_loss_plot(self, loss, name=None):
        plt.title(str(name))
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.axis([0, len(loss), 0, 20])
        plt.plot(loss)
        plt.savefig(str(name) + ".png")
        plt.close()

    def costFunction(self, prediction, actual):
        # prediction is a 1x1 matrix
        # actual is a 1x1 matrix
        # print(prediction.shape, actual)
        result = actual @ np.log(prediction) + (1 - actual) @ np.log(1 - prediction)
        return -np.sum(result) / len(actual)


    # sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x) + 1e-8) # adding a small number to remove overflow
    
    def dSigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))


    def train(self, X, Y):
        # X is 630x2
        # Y is 630x1
        
        for i in range(self.epochs):
            # Feedforward - input to hidden
            hiddenLayer = X[i, :] @ self.weights_1 + self.bias_1 # 1x150
            activatedHiddenLayer = self.sigmoid(hiddenLayer) # 1x150

            # Feedforward - hidden to output
            outputLayer = activatedHiddenLayer @ self.weights_2 + self.bias_2 # 1x1
            activatedOutputLayer = self.sigmoid(outputLayer) # 1x1

            # Tracking train loss
            self.train_loss.append(self.costFunction(activatedOutputLayer, Y[i, :]))

            # For backpropation, we have to use gradient of cross entropy loss

            # Backpropagation - output to hidden
            dBias_2 = activatedOutputLayer - Y[i, :] # 1x1
            dWeights_2 = activatedHiddenLayer.T @ (activatedOutputLayer - Y[i, :]) # 150x1

            # Backpropagation - hidden to input
            dBias_1 = ((self.weights_2 @ dBias_2) * self.dSigmoid(hiddenLayer).T).T # 150x1
            dWeights_1 = ((self.weights_2 @ dBias_2) @ X[i, :].reshape((1, 2)) * self.dSigmoid(hiddenLayer).T).T # 2x150

            # Update weights and bias
            self.weights_2 -= self.learningRate * dWeights_2
            self.bias_2 -= self.learningRate * dBias_2
            self.weights_1 -= self.learningRate * dWeights_1
            self.bias_1 -= self.learningRate * dBias_1
    
    def test(self, X, Y):
        # Feedforward - input to hidden
        for i in range(self.epochs):
            # Feedforward - input to hidden
            hiddenLayer = X[i, :] @ self.weights_1 + self.bias_1
            activatedHiddenLayer = self.sigmoid(hiddenLayer)

            # Feedforward - hidden to output
            outputLayer = activatedHiddenLayer @ self.weights_2 + self.bias_2
            activatedOutputLayer = self.sigmoid(outputLayer)
            
            # Tracking test loss
            self.test_loss.append(self.costFunction(activatedOutputLayer, Y[i, :]))


nn = NeuralNetwork(150, 0.01)
trainX, trainY, testX, testY, grid = nn.load_data()
nn.train(trainX, trainY)
nn.test(testX, testY)
nn.save_loss_plot(nn.train_loss, "Train loss")
nn.save_loss_plot(nn.test_loss, "Test loss")

'''
Reference:
General structure of neural network: https://www.youtube.com/watch?v=aircAruvnKk
I got to know that cross-entropy function is good for classification problems: https://www.analyticsvidhya.com/blog/2021/02/cost-function-is-no-rocket-science/
Cross-entropy loss function function formule: https://eng.libretexts.org/Bookshelves/Computer_Science/Applied_Programming/Book%3A_Neural_Networks_and_Deep_Learning_(Nielsen)/03%3A_Improving_the_way_neural_networks_learn/3.01%3A_The_cross-entropy_cost_function#:~:text=We%20define%20the%20cross%2Dentropy,is%20the%20corresponding%20desired%20output.
'''