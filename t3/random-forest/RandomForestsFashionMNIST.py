import data_loader
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

x_train, y_train, x_test, y_test = data_loader.getFashionMNIST()
x_train = x_train.reshape(x_train.shape[0], 28 * 28)
x_test = x_test.reshape(x_test.shape[0], 28 * 28)

start_time = time.time()

forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
forest.fit(x_train, y_train.ravel())

end_time = time.time()

prediction = forest.predict(x_test)

print("Random Trees - Fashion MNIST accuracy:", accuracy_score(y_test, prediction) * 100)
print("Program trained for", end_time - start_time, "seconds")
