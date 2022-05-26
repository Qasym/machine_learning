import numpy as np
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

# Initialize the weights
weights1 = np.random.rand(2, 150)
weights2 = np.random.rand(150, 1)

def neural_network(X, Y):
    pass

def load_data():
    # Load data
    train = np.loadtxt('Trn.csv', delimiter=',')
    test = np.loadtxt('Tst.csv', delimiter=',')
    grid = np.loadtxt('Grid.csv', delimiter=',')

    # Obtain training data
    trainX = train[:, :2]
    trainY = train[:, 2]

    # Obtain test data
    testX = test[:, :2]
    testY = test[:, 2]

    # We don't need to separate grid data, since it is only X without Y

    return trainX, trainY, testX, testY, grid

trainX, trainY, testX, testY, grid = load_data()

print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)
print(grid.shape)