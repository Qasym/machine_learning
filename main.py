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

def costFunction(prediction, actual):
    # X is 630x1 & Y is 630x1
    result = actual @ np.log(prediction) + (1 - actual) @ np.log(1 - prediction)
    return -np.sum(result) / len(actual)

# sigmoid function
def activation(x):
    return 1 / (1 + np.exp(-x) + 1e-8) # adding a small number to remove overflow

def neural_network(X, Y, learningRate):
    # X is 630x2
    # Y is 630x1
    
    # Feedforward - input to hidden
    X = np.insert(X, 2, 1, axis=1) # insert bias at the end of each column
    param1 = np.insert(weights1, 2, 1, axis=0) # insert bias at the end of each row
    hiddenLayer = activation(X @ param1) # 630x150

    # Feedforward - hidden to output
    hiddenLayer = np.insert(hiddenLayer, 2, 1, axis=1) # insert bias at the end of each column
    param2 = np.insert(weights2, 2, 1, axis=0) # insert bias at the end of each row
    outputLayer = activation(hiddenLayer @ param2) # 630x1

    # Cost function - cross-entropy cost function
    print(costFunction(outputLayer, Y))
    

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

neural_network(trainX, trainY, 0.001)


'''
Reference:
General structure of neural network: https://www.youtube.com/watch?v=aircAruvnKk
I got to know that cross-entropy function is good for classification problems: https://www.analyticsvidhya.com/blog/2021/02/cost-function-is-no-rocket-science/
Cross-entropy loss function function formule: https://eng.libretexts.org/Bookshelves/Computer_Science/Applied_Programming/Book%3A_Neural_Networks_and_Deep_Learning_(Nielsen)/03%3A_Improving_the_way_neural_networks_learn/3.01%3A_The_cross-entropy_cost_function#:~:text=We%20define%20the%20cross%2Dentropy,is%20the%20corresponding%20desired%20output.
'''