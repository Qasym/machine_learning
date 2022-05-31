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



    def __init__(self, hiddenLayerNodes, learningRate, epochs=1):
        # Assign parameters
        self.learningRate = learningRate
        self.epochs = epochs

        # Initialize the weights
        self.weights_1 = np.random.rand(2, hiddenLayerNodes) # from input to hidden
        self.weights_2 = np.random.rand(hiddenLayerNodes, 1) # from hidden to output

        # Initialize the bias
        self.bias_1 = np.random.rand(1, hiddenLayerNodes) # from input to hidden
        self.bias_2 = np.random.rand(1, 1) # from hidden to output

        # Loss & accuracy over epochs tracking for training set
        self.train_loss = []
        self.train_accuracy = []

        # Loss & accuracy over epochs tracking for test set
        self.test_loss = []
        self.test_accuracy = []



    def load_data(self):
        # Load data
        train = np.loadtxt('Trn.csv', delimiter=',')
        test = np.loadtxt('Tst.csv', delimiter=',')
        grid = np.loadtxt('Grid.csv', delimiter=',')

        # Obtain training data
        trainX = train[:, :2]
        trainY = train[:, 2]
        trainY = trainY.reshape(len(trainY), 1)

        # Obtain test data
        testX = test[:, :2]
        testY = test[:, 2]
        testY = testY.reshape(len(testY), 1)

        # We don't need to separate grid data, since it is only X without Y

        return trainX, trainY, testX, testY, grid



    def save_results(self, filename):
        if self.epochs == 1:
            # graphs look awful with epoch, so I just print
            print("Training loss within the only epoch:", "%.2f"%self.train_loss[0])
            print("Training accuracy within the only epoch:", "%.2f"%self.train_accuracy[0])
        else:
            # plotting loss
            plt.title("Loss over epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.plot(self.train_loss, label="Training loss")
            plt.plot(self.test_loss, label="Test loss")
            plt.savefig(filename + "_loss.png")
            plt.close()

            # plotting accuracy
            plt.title("Accuracy over epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.plot(self.train_accuracy, label="Training accuracy")
            plt.plot(self.test_accuracy, label="Test accuracy")
            plt.savefig(filename + "_accuracy.png")
            plt.close()



    def costFunction(self, prediction, actual):
        # prediction is a 1x1 matrix
        # actual is a 1x1 matrix
        result = actual * np.log(prediction) + (1 - actual) * np.log(1 - prediction)
        return -np.sum(result) / len(actual)



    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x) + 1e-8) # adding a small number to remove overflow
    


    def dSigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))



    def feedforward(self, X):
        # Feedforward - input to hidden
        hiddenLayer = X @ self.weights_1 + self.bias_1 # 1x150
        activatedHiddenLayer = self.sigmoid(hiddenLayer) # 1x150

        # Feedforward - hidden to output
        outputLayer = activatedHiddenLayer @ self.weights_2 + self.bias_2 # 1x1
        activatedOutputLayer = self.sigmoid(outputLayer) # 1x1

        return activatedOutputLayer, activatedHiddenLayer, hiddenLayer



    def backpropation(self, input, prediction, actual, aHL, hL):
        # For backpropation, we have to use gradient of cross entropy loss

        # Backpropagation - output to hidden
        dBias_2 = prediction - actual # 1x1
        dWeights_2 = aHL.T @ (prediction - actual) # 150x1

        # Backpropagation - hidden to input
        dBias_1 = ((self.weights_2 @ dBias_2) * self.dSigmoid(hL).T).T # 150x1
        dWeights_1 = ((self.weights_2 @ dBias_2) @ X[i, :].reshape((1, 2)) * self.dSigmoid(hL).T).T # 2x150

        return dBias_1, dWeights_1, dBias_2, dWeights_2



    def updateWeightsBiases(self, dB1, dW1, dB2, dW2):
        # Update weights and bias
        self.weights_2 -= self.learningRate * dW2
        self.bias_2 -= self.learningRate * dB2
        self.weights_1 -= self.learningRate * dW1
        self.bias_1 -= self.learningRate * dB1
            


    def SGD(self, X, Y):
        # Fully stochastic gradient descent
        for i in range(len(X)):
            # aHL - activatedHiddenLayer
            # hL - hiddenLayer
            prediction, aHL, hL = self.feedforward(X[i])
            dB1, dW1, dB2, dW2 = self.backpropation(prediction, X[i], Y[i], aHL, hL)
            self.updateWeightsBiases(dB1, dW1, dB2, dW2)



    def getAccuracyLoss(self, X, Y):
        # Get prediction
        prediction, a, b = self.feedforward(X)

        # Calculate loss
        loss = self.costFunction(prediction, Y)

        # Calculate accuracy
        accuracy = ((prediction >= 0.5).astype(int) == Y).astype(int).sum() / len(Y)
        accuracy *= 100
        return accuracy, loss



    def updateTrainAccuracyLoss(self, X, Y):
        # Accuracy and loss tracking
        accuracy, loss = self.getAccuracyLoss(X, Y)
        self.train_accuracy.append(accuracy)
        self.train_loss.append(loss)



    def train(self, X, Y):
        for epoch in range(self.epochs):
            self.updateTrainAccuracyLoss(X, Y) # initial accuracy and loss
            self.SGD(X, Y)
            
        # Accuracy and loss of the last epoch
        self.updateTrainAccuracyLoss(X, Y)
        print("\n\n\n\nTraining finished")
        print("Training loss:", "%.2f"%self.train_loss[-1])
        print("Training accuracy:", "%.2f"%self.train_accuracy[-1])
        print("\n\n\n\n")

    
    def test(self, X, Y):
        accuracy, loss = self.getAccuracyLoss(X, Y)
        
        # Print message after finish
        print("\n\n\n\nTesting finished")
        print("Test loss:", "%.2f"%loss)
        print("Test accuracy: ", "%.2f"%accuracy, "%", sep="")
        print("\n\n\n\n")



    def updateTestAccuracyLoss(self, X, Y):
        # Accuracy and loss tracking
        accuracy, loss = self.getAccuracyLoss(X, Y)
        self.test_accuracy.append(accuracy)
        self.test_loss.append(loss)



    def testAsTrain(self, trainX, trainY, testX, testY):
        # Train the model
        for epoch in range(self.epochs):
            self.updateTrainAccuracyLoss(trainX, trainY) # initial accuracy and loss
            self.updateTestAccuracyLoss(testX, testY)
            self.SGD(trainX, trainY)

        # Accuracy and loss of the last epoch
        self.updateTrainAccuracyLoss(trainX, trainY)
        self.updateTestAccuracyLoss(testX, testY)
        



    def justPrediction(self, X):
        prediction, a, b = self.feedforward(X)

        prediction = (prediction >= 0.5).astype(int)

        return prediction



    def plotGrid(self, gridX, gridY, name="grid"):
        x = np.asarray(gridX[:, 0])
        y = np.asarray(gridX[:, 1])
        z = np.where(gridY[:, 0] == 1, 'r', 'b')

        plt.title("Grid")
        plt.scatter(x, y, c=z)
        plt.savefig(name + ".png")
        plt.close()



def main():
    nn40 = NeuralNetwork(hiddenLayerNodes=250, learningRate=0.1, epochs=500)
    trainX, trainY, testX, testY, grid = nn40.load_data()
    nn40.train(trainX[:40, :], trainY[:40, :])

    # Showing results
    print("\n\n\n-- -- -- -- -- -- -- -- -- --")
    print("For the first 40 data points")
    nn40.save_train_results("40")
    nn40.test(testX, testY)
    print("-- -- -- -- -- -- -- -- -- --\n\n\n")

    grid40 = nn40.justPrediction(grid)
    nn40.plotGrid(grid, grid40, "grid40")

    nnAll = NeuralNetwork(hiddenLayerNodes=250, learningRate=0.1, epochs=500)
    nnAll.train(trainX, trainY)

    # Showing results
    print("\n\n\n-- -- -- -- -- -- -- -- -- --")
    print("For the whole train set")
    nnAll.save_train_results("all")
    nnAll.test(testX, testY)
    print("-- -- -- -- -- -- -- -- -- --\n\n\n")

    gridAll = nnAll.justPrediction(grid)
    nnAll.plotGrid(grid, gridAll, "gridAll")

main()