import data_loader
import time
import tensorflow as tf
import pandas as pd

x_train, y_train, x_test, y_test = data_loader.getFashionMNIST()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, (3,3), padding='same', activation="relu",input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Dropout(rate=0.2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(200, activation="relu"),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(10, activation="softmax")
    ]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, min_delta=0.001)
]

start_time = time.time()

model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

history = model.fit(x_train,y_train, 
                    epochs=20,validation_data=(x_test,y_test),
                    callbacks=callbacks)

end_time = time.time()

metrics = pd.DataFrame(history.history)
metrics[['val_accuracy', 'accuracy']].plot().get_figure().savefig('opt_FashionMNIST_metrics.png')

print("Program trained for", end_time - start_time, "seconds")

# Reference to build cnn: https://cnvrg.io/cnn-tensorflow/