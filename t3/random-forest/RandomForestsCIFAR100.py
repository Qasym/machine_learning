import data_loader
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

x_train, y_train, x_test, y_test = data_loader.getCIFAR100()
x_train = x_train.reshape(x_train.shape[0], 32 * 32 * 3)
x_test = x_test.reshape(x_test.shape[0], 32 * 32 * 3)

start_time = time.time()

forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1, verbose=0, 
                                max_depth=15)
forest.fit(x_train, y_train.ravel())

end_time = time.time()

prediction = forest.predict(x_test)
# print("prediction.shape", prediction.shape)
# print("y_test.shape", y_test.shape)

print("Random Forests - CIFAR 100 accuracy:", accuracy_score(y_test, prediction) * 100)
print("Program trained for", end_time - start_time, "seconds")

print("##########")
print("forest.score(train_x, train_y) =", forest.score(x_train, y_train.ravel()))
print("forest.score(test_x, test_y) =", forest.score(x_test, y_test.ravel()))
