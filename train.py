import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

path = '/homes/es314/cv/mnist/mnist.npz'
f = np.load(path)
x_train, y_train =  f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
f.close()


def create_nerual_network():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # Simple Dense Layer
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # Simple Dense Layer
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))   # Output layer

    return model
model = create_nerual_network()


x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, verbose=2)
train_loss, train_acc = model.evaluate(x_train, y_train)

x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Evaluationn loss:", test_loss)
print("Evaluation accuracy:", test_accuracy)


model.save('model_digits.model')



