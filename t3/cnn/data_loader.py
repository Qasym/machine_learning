import keras

def getMNIST():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    return x_train, y_train, x_test, y_test

def getCIFAR100():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    x_train = x_train.astype('float32') / 255 # normalize to [0,1]
    # x_train = x_train.reshape(x_train.shape[0], 32 * 32 * 3) # reshape 

    x_test = x_test.astype('float32') / 255 # normalize to [0,1]
    # x_test = x_test.reshape(x_test.shape[0], 32 * 32 * 3) # reshape


    # y_train = keras.utils.to_categorical(y_train, 100)
    # y_test = keras.utils.to_categorical(y_test, 100)

    return x_train, y_train, x_test, y_test

def getFashionMNIST():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255 # normalize to [0,1]
    # x_train = x_train.reshape(x_train.shape[0], 28 * 28) # reshape

    x_test = x_test.astype('float32') / 255 # normalize to [0,1]
    # x_test = x_test.reshape(x_test.shape[0], 28 * 28) # reshape

    # y_train = keras.utils.to_categorical(y_train, 10)
    # y_test = keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test
