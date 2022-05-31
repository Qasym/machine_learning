import numpy as np
import matplotlib.pyplot as plt

'''
input layer consists of 2 neurons
hidden layer consists of hiddenLayerNodes neurons
so the weights1 variable that is between input layer
and hidden layer should be 2xhiddenLayerNodes
since my design implies one hidden layer
and we have only binary output (either 0 or 1)
weights2 variable should be hiddenLayerNodesx1
weights2 is a matrix between the output layer and the hidden layer
'''

class NeuralNetwork:
    def __init__(self, hiddenLayerNodes, learningRate, epochs=1, data_entries=None):
        # Assign parameters
        self.learningRate = learningRate
        self.data_entries = data_entries
        self.epochs = epochs

        # Initialize the weights
        self.weights_1 = np.random.rand(2, hiddenLayerNodes) # from input to hidden
        self.weights_2 = np.random.rand(hiddenLayerNodes, 1) # from hidden to output

        # Initialize the bias
        self.bias_1 = np.random.rand(1, hiddenLayerNodes) # from input to hidden
        self.bias_2 = np.random.rand(1, 1) # from hidden to output

        # Loss over epochs tracking
        self.train_loss = []
        self.train_accuracy = []

    def load_data(self):
        # Load data
        train = np.loadtxt('Trn.csv', delimiter=',')
        test = np.loadtxt('Tst.csv', delimiter=',')
        grid = np.loadtxt('Grid.csv', delimiter=',')

        # Obtain training data
        trainX = train[:, :2]
        trainY = train[:, 2]
        trainY = trainY.reshape(len(trainY), 1)

        # Set data_entries
        if self.data_entries == None or self.data_entries > len(trainX):
            self.data_entries = len(trainX)

        # Obtain test data
        testX = test[:, :2]
        testY = test[:, 2]
        testY = testY.reshape(len(testY), 1)

        # We don't need to separate grid data, since it is only X without Y

        return trainX, trainY, testX, testY, grid

    def save_train_results(self, name=None):
        if self.epochs == 1:
            # graphs look awful with epoch, so I just print
            print("Training loss within the only epoch:", "%.2f"%self.train_loss[0])
            print("Training accuracy within the only epoch:", "%.2f"%self.train_accuracy[0])
        else:
            # plotting loss
            plt.title("Train loss over epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            # plt.xticks(np.arange(1, self.epochs + 2, 1))
            plt.plot(self.train_loss)
            if name == None:
                plt.savefig("train_loss.png")
            else:
                plt.savefig("train_loss_" + name + ".png")
            plt.close()

            # plotting accuracy
            plt.title("Train accuracy over epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            # plt.xticks(np.arange(1, self.epochs + 2, 1))
            plt.plot(self.train_accuracy)
            if name == None:
                plt.savefig("train_accuracy.png")
            else:
                plt.savefig("train_accuracy_" + name + ".png")
            plt.close()

            # printing the last accuracy and loss
            print("Training loss:", "%.2f"%self.train_loss[-1])
            print("Training accuracy:", "%.2f"%self.train_accuracy[-1])

    def costFunction(self, prediction, actual):
        # prediction is a 1x1 matrix
        # actual is a 1x1 matrix
        # print(prediction.shape, actual)
        result = actual * np.log(prediction) + (1 - actual) * np.log(1 - prediction)
        return -np.sum(result) / len(actual)


    # sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x) + 1e-8) # adding a small number to remove overflow
    
    def dSigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def getAccuracyLoss(self, X, Y):
        # Feedforward - input to hidden
        hiddenLayer = X @ self.weights_1 + self.bias_1 # 630x150
        activatedHiddenLayer = self.sigmoid(hiddenLayer) # 630x150

        # Feedforward - hidden to output
        outputLayer = activatedHiddenLayer @ self.weights_2 + self.bias_2 # 630x1
        activatedOutputLayer = self.sigmoid(outputLayer) # 630x1

        # Calculate loss
        loss = self.costFunction(activatedOutputLayer, Y)

        # Calculate accuracy
        accuracy = ((activatedOutputLayer >= 0.5).astype(int) == Y).astype(int).sum() / len(Y)
        accuracy *= 100
        return accuracy, loss

    def train(self, X, Y):
        # X is 630x2
        # Y is 630x1
        for epoch in range(self.epochs):
            # Accuracy and lost tracking
            accuracy, loss = self.getAccuracyLoss(X, Y)
            self.train_accuracy.append(accuracy)
            self.train_loss.append(loss)
            
            # Fully stochastic gradient descent
            for i in range(self.data_entries):
                # Feedforward - input to hidden
                hiddenLayer = X[i, :] @ self.weights_1 + self.bias_1 # 1x150
                activatedHiddenLayer = self.sigmoid(hiddenLayer) # 1x150

                # Feedforward - hidden to output
                outputLayer = activatedHiddenLayer @ self.weights_2 + self.bias_2 # 1x1
                activatedOutputLayer = self.sigmoid(outputLayer) # 1x1

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
            
        # Accuracy and loss of the last epoch
        accuracy, loss = self.getAccuracyLoss(X, Y)
        self.train_accuracy.append(accuracy)
        self.train_loss.append(loss)

    
    def test(self, X, Y):
        accuracy, loss = self.getAccuracyLoss(X, Y)
        
        # Print message after finish
        print("Test loss:", "%.2f"%loss)
        print("Test accuracy: ", "%.2f"%accuracy, "%", sep="")

    def justPrediction(self, X):
        # Feedforward - input to hidden
        hiddenLayer = X @ self.weights_1 + self.bias_1 # 630x150
        activatedHiddenLayer = self.sigmoid(hiddenLayer) # 630x150

        # Feedforward - hidden to output
        outputLayer = activatedHiddenLayer @ self.weights_2 + self.bias_2 # 630x1
        activatedOutputLayer = self.sigmoid(outputLayer) # 630x1

        activatedOutputLayer = (activatedOutputLayer >= 0.5).astype(int)

        return activatedOutputLayer

    def plotGrid(self, gridX, gridY):
        x = np.asarray(gridX[:, 0])
        y = np.asarray(gridX[:, 1])
        z = np.where(gridY[:, 0] == 1, 'r', 'b')

        plt.title("Grid")
        plt.scatter(x, y, c=z)
        plt.savefig("grid.png")
        plt.close()


# nn = NeuralNetwork(hiddenLayerNodes=250, learningRate=0.01, epochs=100)
# trainX, trainY, testX, testY, grid = nn.load_data()
# nn.train(trainX, trainY)
# nn.test(testX, testY)
# nn.save_train_results()

nn40 = NeuralNetwork(hiddenLayerNodes=250, learningRate=0.01, epochs=100, data_entries=40)
trainX, trainY, testX, testY, grid = nn40.load_data()
nn40.train(trainX, trainY)

# Showing results
# print("-- -- -- -- -- -- -- -- -- --")
# print("For the first 40 data points")
# nn40.save_train_results("40")
# nn40.test(testX, testY)
# print("-- -- -- -- -- -- -- -- -- --", end="\n\n")

grid40 = nn40.justPrediction(grid)
nn40.plotGrid(grid, grid40)
