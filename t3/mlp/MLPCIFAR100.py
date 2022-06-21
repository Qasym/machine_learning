import data_loader
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

x_train, y_train, x_test, y_test = data_loader.getCIFAR100()
x_train = x_train.reshape(x_train.shape[0], 32 * 32 *3)
x_test = x_test.reshape(x_test.shape[0], 32 * 32 * 3)

start_time = time.time()

model = MLPClassifier(hidden_layer_sizes=(1000, 1000), alpha=0.001, early_stopping=True,
                      learning_rate='constant', activation='logistic', verbose=1)
model.fit(x_train, y_train.ravel())

end_time = time.time()

prediction = model.predict(x_test)

print("MLP - CIFAR 100 accuracy:", accuracy_score(y_test, prediction) * 100, "%")
print("Program trained for", end_time - start_time, "seconds")

print("#########")
print("train accuracy:", model.score(x_train, y_train.ravel()) * 100, "%")
print("test accuracy:", model.score(x_test, y_test.ravel()) * 100, "%")
