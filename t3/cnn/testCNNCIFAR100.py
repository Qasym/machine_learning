import data_loader
import tensorflow as tf
import keras
from sklearn.metrics import accuracy_score

model = keras.models.load_model('cnn_cifar100.h5')

x_train, y_train, x_test, y_test = data_loader.getCIFAR100()

prediction = model.predict(x_test)

# print("x_test:", x_test.shape)
# print("y_test:", y_test.shape)

acc, loss = model.evaluate(x_test, y_test, verbose=2)
print("test acc:", acc)
print("test loss:", loss)
