import numpy as np
import sys

#Задача: принять решение, сходить ли в киберспортивный клуб
#Имеем 3 входа
#1ый вход: есть ли настроение
#2ой вход: есть ли деньги
#3ий вход: есть ли с кем сходить


class Cybersportclub(object):

    def __init__(self, learning_rate=0.1):
        self.weights_0_1 = np.random.normal(0.0, 2 ** -0.5, (2, 3))
        self.weights_1_2 = np.random.normal(0.0, 1, (1, 2))
        self.sigmoid_mapper = np.vectorize(self.sigmoid)
        self.learning_rate = np.array([learning_rate])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        inputs_1 = np.dot(self.weights_0_1, inputs)
        outputs_1 = self.sigmoid_mapper(inputs_1)

        inputs_2 = np.dot(self.weights_1_2, outputs_1)
        outputs_2 = self.sigmoid_mapper(inputs_2)
        return outputs_2

    def train(self, inputs, expected_predict):
        inputs_1 = np.dot(self.weights_0_1, inputs)
        outputs_1 = self.sigmoid_mapper(inputs_1)

        inputs_2 = np.dot(self.weights_1_2, outputs_1)
        outputs_2 = self.sigmoid_mapper(inputs_2)
        actual_predict = outputs_2[0]

        error_layer_2 = np.array([actual_predict - expected_predict])
        gradient_layer_2 = actual_predict * (1 - actual_predict)
        weights_delta_layer_2 = error_layer_2 * gradient_layer_2
        self.weights_1_2 -= (np.dot(weights_delta_layer_2, outputs_1.reshape(1, len(outputs_1)))) * self.learning_rate

        error_layer_1 = weights_delta_layer_2 * self.weights_1_2
        gradient_layer_1 = outputs_1 * (1 - outputs_1)
        weights_delta_layer_1 = error_layer_1 * gradient_layer_1
        self.weights_0_1 -= np.dot(inputs.reshape(len(inputs), 1), weights_delta_layer_1).T * self.learning_rate

    def real(self, inputs):
        inputs_1 = np.dot(self.weights_0_1, inputs)
        outputs_1 = self.sigmoid_mapper(inputs_1)

        inputs_2 = np.dot(self.weights_1_2, outputs_1)
        outputs_2 = self.sigmoid_mapper(inputs_2)
        return outputs_2


def MSE(y, Y):
    return np.mean((y-Y)**2)


train = [
    ([0, 0, 0], 0),
    ([0, 0, 1], 0),
    ([0, 1, 0], 0),
    ([0, 1, 1], 0),
    ([1, 0, 0], 0),
    ([1, 0, 1], 1),
    ([1, 1, 0], 1),
    ([1, 1, 1], 1),
]


real = [
    [0.4, 0.2, 0.3],
    [0.2, 0.3, 0.5],
    [0.1, 0.76, 0.44],
    [0.34, 0.866, 0.91],
    [0.88, 0.2, 0.113],
    [0.788, 0.43, 0.899],
    [0.832, 0.688, 0.01],
    [0.997, 0.887, 0.671],
    [0.668, 0.235, 0.34],
    [0.781, 0.969, 0.11],
    [0.561, 0.647, 0.464],
    [0.355, 0.8912, 0.755],
    [0.8974, 0.8231, 0.06],
    [0.364, 0.2156, 0.699],
    [0.2256, 0.741, 0.148],
    [0.648, 0.256, 0.6789],
    [0.985, 0.6742, 0.962],
    [0.8745, 0.2654, 0.135],
    [0.1235, 0.478, 0.2314],
    [0.96, 0.63, 0.745],
    [0.23, 0.12, 0.069],
    [0.639, 0.8756, 0.91],
    [0.92, 0.1566, 0.693],
    [0.745, 0.2356, 0.1487],
    [0.326, 0.65, 0.98],
    [0.123, 0.879, 0.756],
    [0.912, 0.857, 0.98],
    [0.293, 0.589, 0.456],
    [0.698, 0.566, 0.745],
    [0.876, 0.9857, 0.656]
]


epochs = 8000
learning_rate = 0.12

network = Cybersportclub(learning_rate=learning_rate)

for e in range(epochs):
    inputs_ = []
    correct_predictions = []
    for input_stat, correct_predict in train:
        network.train(np.array(input_stat), correct_predict)
        inputs_.append(np.array(input_stat))
        correct_predictions.append(np.array(correct_predict))

    train_loss = MSE(network.predict(np.array(inputs_).T), np.array(correct_predictions))
    sys.stdout.write("\rProgress: {}, Training loss: {}".format(str(100 * e/float(epochs))[:4], str(train_loss)[:5]))


for input_stat, correct_predict in train:
    print("For input: {} the prediction is: {}, expected: {}".format(
        str(input_stat),
        str(network.predict(np.array(input_stat)) > .5),
        str(correct_predict == 1)))


for input_stat, correct_predict in train:
    print("For input: {} the prediction is: {}, expected: {}".format(
        str(input_stat),
        str(network.predict(np.array(input_stat))),
        str(correct_predict)))

for e in range(epochs):
    inputs_ = []
    correct_predictions = []
    for input_stat in real:
        network.real(np.array(input_stat))
        inputs_.append(np.array(input_stat))

for input_stat in real:
    print("For input: {} the prediction is: {}, boolean: {}".format(
        str(input_stat),
        str(network.predict(np.array(input_stat))),
        str(network.predict(np.array(input_stat)) > .5)))



