import gzip
import os
import urllib.parse
import urllib.request

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

from autograd import exp, gaussian_sample, log, maximum, tanh, tensor
from autograd.optimizer import RMSProp

EPOCHS = 200
BATCH_SIZE = 128

LOCAL_PATH = './datasets/mnist/'
REMOTE_PATH = 'http://yann.lecun.com/exdb/mnist/'

FPS = 50
DURATION = 10


def read_dataset(name):
    path = os.path.join(LOCAL_PATH, name)
    if os.path.isfile(path):
        with open(path, 'rb') as file:
            data = file.read()
    else:
        file_url = urllib.parse.urljoin(REMOTE_PATH, name)
        with urllib.request.urlopen(file_url) as remote_file, \
                open(path, 'wb') as file:
            data = remote_file.read()
            file.write(data)

    decompressed_data = gzip.decompress(data)
    return np.frombuffer(decompressed_data, dtype=np.uint8).copy()


class Dataset:

    def __init__(self, X, batch_size=128):
        self._X = X
        self._batch_size = batch_size

    def __iter__(self):
        np.random.shuffle(self._X)
        return iter(np.array_split(self._X, self._X.shape[0]
                                   // self._batch_size))


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


class Encoder:

    def __init__(self):
        self._input_layer = Linear(784, 64)
        self._middle_layer = Linear(64, 32)
        self._mean_layer = Linear(32, 2)
        self._log_standard_deviation_layer = Linear(32, 2)

    def __call__(self, x):
        h = tanh(self._input_layer(x))
        h = tanh(self._middle_layer(h))
        mean = self._mean_layer(h)
        standard_deviation = exp(self._log_standard_deviation_layer(h))

        return mean, standard_deviation

    @property
    def parameters(self):
        return self._input_layer.parameters \
            + self._middle_layer.parameters \
            + self._mean_layer.parameters \
            + self._log_standard_deviation_layer.parameters

    def save(self, path):
        with open(path, 'wb') as file:
            values = tuple(parameter.data for parameter in self.parameters)
            np.savez(file, *values)

    def load(self, path):
        with open(path, 'rb') as file:
            for parameter, value in zip(self.parameters,
                                        np.load(file).values()):
                parameter.data = value

class Decoder:

    def __init__(self):
        self._input_layer = Linear(2, 32)
        self._middle_layer = Linear(32, 64)
        self._mean_layer = Linear(64, 784)
        self._log_standard_deviation_layer = Linear(64, 784)

    def __call__(self, z):
        h = tanh(self._input_layer(z))
        h = tanh(self._middle_layer(h))
        mean = self._mean_layer(h)
        standard_deviation = exp(self._log_standard_deviation_layer(h))

        return mean, standard_deviation

    @property
    def parameters(self):
        return self._input_layer.parameters \
            + self._middle_layer.parameters \
            + self._mean_layer.parameters \
            + self._log_standard_deviation_layer.parameters

    def save(self, path):
        with open(path, 'wb') as file:
            values = tuple(parameter.data for parameter in self.parameters)
            np.savez(file, *values)

    def load(self, path):
        with open(path, 'rb') as file:
            for parameter, value in zip(self.parameters,
                                        np.load(file).values()):
                parameter.data = value


def compute_reconstruction_loss(x, mean, standard_deviation, epsilon=1e-6):
    """Compute gaussian negative log likelihood loss."""
    return (log(maximum(standard_deviation, epsilon))
        + (mean - x)**2.0 / maximum(standard_deviation, epsilon)) \
            .mean(axis=-2) / 2.0

def compute_regularization_loss(mean, standard_deviation):
    """Compute Kullback-Leibler divergence between gaussian distribution and
    standard normal distribution.
    """
    return -(1.0 + log(standard_deviation**2.0) - mean**2.0
        - standard_deviation**2.0).sum(axis=-2) / 2.0


if __name__ == '__main__':
    # prepare dataset
    X = read_dataset('train-images-idx3-ubyte.gz')[0x10:].reshape(-1, 28, 28)
    Y = read_dataset('train-labels-idx1-ubyte.gz')[8:]

    X = X[Y == 9].reshape(-1, 28*28, 1).astype(np.float32) / 255.0

    dataset = Dataset(X, batch_size=BATCH_SIZE)

    # train models
    encoder = Encoder()
    decoder = Decoder()

    optimizer = RMSProp(encoder.parameters + decoder.parameters,
                        learning_rate=1e-3, weight_decay=1e-3)

    for epoch in range(EPOCHS):
        for batch, x in enumerate(dataset):
            mean_z, standard_deviation_z = encoder(x)
            z = gaussian_sample(mean_z, standard_deviation_z)
            mean_x, standard_deviation_x = decoder(z)

            reconstruction_loss = compute_reconstruction_loss(
                x, mean_x, standard_deviation_x)
            regularization_loss = compute_regularization_loss(
                mean_z, standard_deviation_z)

            loss = (reconstruction_loss + regularization_loss).mean()

            loss.backpropagate()

            optimizer.step()

        print(f'epoch {epoch + 1} / {EPOCHS}, loss = {loss.data:.3f}')

    encoder.save('./models/vae/encoder.npz')
    decoder.save('./models/vae/decoder.npz')

    # create animation
    theta = np.linspace(0.0, 2.0*np.pi, num=DURATION*FPS)
    latent_codes = \
        np.expand_dims(
            np.stack([
                np.cos(theta),
                np.sin(theta)
            ], axis=-1),
            axis=-1)

    px = 1.0 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(640*px, 640*px))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    images, _ = decoder(latent_codes)

    images = images.data.reshape(-1, 28, 28)
    images = np.clip(images, 0.0, 1.0)
    im = ax.imshow(images[0], cmap='gray')

    def animate(x):
        im.set_data(x)

        return im,

    animation.FuncAnimation(
        fig, animate, images, interval=int(1e3 / FPS), blit=True) \
            .save('vae_demo.gif', fps=FPS)