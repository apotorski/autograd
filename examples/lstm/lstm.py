import itertools

import numpy as np
from matplotlib import pyplot as plt

from autograd import mean_squared_error_loss, sigmoid, stack, tanh, tensor
from autograd.optimizer import RMSProp

plt.style.use('bmh')


SEQUENCE_LENGTH = 128
BATCH_SIZE = 1

ITERATIONS = 5000

LOG_PERIOD = 20


class LSTM:

    class LSTMCell:

        class Linear:

            def __init__(self, input_features, output_features):
                # Xavier initialization
                bound = 1.0 / np.sqrt(input_features)
                self._W = tensor(np.random.uniform(
                    -bound, bound, (output_features, input_features)))
                self._b = tensor(np.random.uniform(
                    -bound, bound, (output_features, 1)))

            def __call__(self, x):
                return self._W @ x + self._b

            @property
            def parameters(self):
                return self._W, self._b

        def __init__(self, input_size, hidden_size):
            self._ii_layer = LSTM.LSTMCell.Linear(input_size, hidden_size)
            self._hi_layer = LSTM.LSTMCell.Linear(hidden_size, hidden_size)

            self._if_layer = LSTM.LSTMCell.Linear(input_size, hidden_size)
            self._hf_layer = LSTM.LSTMCell.Linear(hidden_size, hidden_size)

            self._ig_layer = LSTM.LSTMCell.Linear(input_size, hidden_size)
            self._hg_layer = LSTM.LSTMCell.Linear(hidden_size, hidden_size)

            self._io_layer = LSTM.LSTMCell.Linear(input_size, hidden_size)
            self._ho_layer = LSTM.LSTMCell.Linear(hidden_size, hidden_size)

        def __call__(self, x, h_0, c_0):
            i = sigmoid(self._ii_layer(x) + self._hi_layer(h_0))

            f = sigmoid(self._if_layer(x) + self._hf_layer(h_0))

            g = tanh(self._ig_layer(x) + self._hg_layer(h_0))

            o = sigmoid(self._io_layer(x) + self._ho_layer(h_0))

            c_1 = f*c_0 + i*g
            h_1 = o*tanh(c_1)

            return h_1, c_1

        @property
        def parameters(self):
            return self._ii_layer.parameters + self._hi_layer.parameters \
                + self._if_layer.parameters + self._hf_layer.parameters \
                + self._ig_layer.parameters + self._hg_layer.parameters \
                + self._io_layer.parameters + self._ho_layer.parameters

    def __init__(self):
        self._layers = (
            LSTM.LSTMCell(1, 32),
            LSTM.LSTMCell(32, 1)
        )

    def __call__(self, x, h_0, c_0):
        h_1, c_1 = [], []

        previous_h, c = self._layers[0](x, h_0[0], c_0[0])
        h_1.append(previous_h)
        c_1.append(c)

        for layer, h, c in zip(self._layers[1:], h_0[1:], c_0[1:]):
            previous_h, c = layer(previous_h, h, c)
            h_1.append(previous_h)
            c_1.append(c)

        return tuple(h_1), tuple(c_1)

    @property
    def parameters(self):
        return tuple(itertools.chain.from_iterable(layer.parameters
                                                   for layer in self._layers))

    def save(self, path):
        with open(path, 'wb') as file:
            values = tuple(parameter.data for parameter in self.parameters)
            np.savez(file, *values)

    def load(self, path):
        with open(path, 'rb') as file:
            for parameter, value in zip(self.parameters,
                                        np.load(file).values()):
                parameter.data = value


if __name__ == '__main__':
    # prepare dataset
    t = np.linspace(0.0, 1.0, SEQUENCE_LENGTH) \
        .reshape(SEQUENCE_LENGTH, BATCH_SIZE, 1, 1)
    omega = 16.0*np.pi
    true_y = np.exp(-t)*np.sin(omega*t)

    # train model
    model = LSTM()

    optimizer = RMSProp(model.parameters, learning_rate=1e-4,
                        weight_decay=1e-3)

    x = np.zeros((BATCH_SIZE, 1, 1))
    for iteration in range(ITERATIONS):
        h = (
            np.zeros((BATCH_SIZE, 32, 1)),
            np.zeros((BATCH_SIZE, 1, 1))
        )
        c = tuple(np.zeros_like(hiddden_state) for hiddden_state in h)

        y = []
        for _ in range(true_y.shape[0]):
            h, c = model(x, h, c)
            y.append(h[-1])
        y = stack(y)

        loss = mean_squared_error_loss(y, true_y)

        loss.backpropagate()

        if (iteration + 1) % LOG_PERIOD == 0:
            print(f'iteration {iteration + 1} / {ITERATIONS}, '
                  f'loss = {loss.data:.6f}')

        optimizer.step()

    model.save('./models/lstm/rnn.npz')

    # plot model's predictions
    t = np.linspace(0.0, 2.0, 2*SEQUENCE_LENGTH)
    true_y = np.exp(-t)*np.sin(omega*t)

    plt.plot(t, true_y, linestyle='--', color='tab:gray', label='True value')

    plt.plot((1.0, 1.0), (-1.5, 1.5), linestyle='--', color='black', lw=1)
    plt.text(0.5, 0.75, 'Training data', ha='center', va='center')
    plt.text(1.5, 0.75, 'Test data', ha='center', va='center')

    x = np.zeros((1, 1))
    h = (
        np.zeros((32, 1)),
        np.zeros((1, 1))
    )
    c = tuple(np.zeros_like(hidden_state) for hidden_state in h)

    y = []
    for _ in range(true_y.shape[0]):
        h, c = model(x, h, c)
        y.append(h[-1])
    y = stack(y)

    y = np.squeeze(y.data)

    plt.plot(t, y, linestyle='-', color='black', label='Predicted value')

    plt.xlim([0.0, 2.0])
    plt.ylim([-1.5, 1.5])
    plt.xlabel('t')
    plt.ylabel('y')

    plt.legend()

    plt.savefig('lstm_demo.png')