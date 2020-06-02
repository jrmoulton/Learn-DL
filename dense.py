import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class Dense:
    def __init__(self, h_layers, n_outputs, img_rows, img_cols, act_funcs=[]):
        self.img_size = img_rows * img_cols
        # Done this way for clarity of user
        self.h_layers = h_layers.append(n_outputs)
        self.n_layers = len(h_layers)
        if act_funcs:
            self.act_funcs = []
            for act_func in act_funcs:
                if act_func == "relu":
                    self.act_funcs.append(self.relu)
                elif act_func == "sigmoid":
                    self.act_funcs.append(self.sigmoid)
                elif act_func == "sofmax":
                    self.act_funcs.append(self.softmax)
        else:
            self.act_funcs = [self.relu for _ in range(self.n_layers)]
            self.act_funcs[-1] = self.sigmoid
        act_func_dict = {self.relu: self.relu_prime,
                         self.sigmoid: self.sigmoid_prime, self.softmax: self.softmax_prime}
        self.act_funcs_prime = []
        for func in self.act_funcs:
            self.act_funcs_prime.append(act_func_dict[func])
        assert len(self.act_funcs) == self.n_layers

        # initialize weights with std. deviation that divides the random initializations by the sqrt of the number of inputs / 2
        # see cs231n Andrej Karpathy time-stamp 49:00 https://youtu.be/gYpoJMlgyXA?t=2942
        self.weights = [np.random.randn(
            self.img_size, h_layers[0]) / np.sqrt(self.img_size/2)]
        for i in range(self.n_layers-1):
            self.weights.append(np.random.randn(
                h_layers[i], h_layers[i+1]) / np.sqrt(h_layers[i] / 2))
        self.weights = np.array(self.weights)
        self.biases = np.array([np.random.randn(y) for y in h_layers])
        self.train_error = []

    def forward(self, inp, train=False):
        zipped = list(zip(self.biases, self.weights))
        if train:
            z = []
            a = [inp]
            for i in range(self.n_layers):
                b, w = zipped[i]
                inp = np.dot(inp, w) + b
                z.append(inp)
                inp = self.act_funcs[i](inp)
                a.append(inp)
            return a, z
        else:
            for i in range(self.n_layers):
                b, w = zipped[i]
                inp = self.act_funcs[i](np.dot(inp, w) + b)
            return inp

    def train(self, data, labels, epochs=500, batch_size=10):
        self.data = data
        self.labels = labels
        self.batch_error = []

        # TODO feed each image and track avg error. Compute gradient and backprop.

        for _ in range(epochs):
            x_batch, y_batch = self._get_batch(batch_size)
            batch_d_b = []
            batch_d_w = []
            for img, label in zip(x_batch, y_batch):
                a, z = self.forward(img, train=True)
                img_d_b = []
                img_d_w = []

                loss = (a[-1] - label) ** 2
                self.batch_error.append(loss.mean())

                # set the final activ. grad. to deriv of loss function evaluated with activation of final layer == result of network
                d_a = (a[-1] - label) * 2
                d_z = 0  # derivaties for current z_nodes
                for i in range(1, len(z)+1):
                    # d_a[-1]
                    # evaluate the derivative of z_nodes using the derivative of the activ. func used, evaluated at zL, and mult. by the leading grad, because of chain rule, that is stored at d_a[i]
                    d_z = self.act_funcs_prime[-i](z[-i]) * d_a

                    # The grad. of biases is just that of z_nodes
                    img_d_b.append(d_z)
                    # TODO There probalby is a numpy operation that could replace this for loop
                    layer_d_w = []
                    for a_l in a[-i-1]:
                        layer_d_w.append(a_l * d_z)
                    img_d_w.append(np.array(layer_d_w))
                    d_a = np.dot(d_z, self.weights[-i].T)
                batch_d_b.append(img_d_b)
                batch_d_w.append(np.array(img_d_w))

            batch_d_w = np.array(batch_d_w).mean(axis=0) * .01
            batch_d_b = np.array(batch_d_b).mean(axis=0) * .01

            self.weights = self.weights - batch_d_w[::-1]
            self.biases = self.biases - batch_d_b[::-1]

            self.train_error.append(np.array(self.batch_error).mean())

    def _get_batch(self, batch_size):
        # indices = np.random.randint(len(self.data), size=batch_size)
        indices = [121, 122]
        return self.data[indices].reshape(batch_size, 784), self.labels[indices]

    def relu(self, inp):
        return np.maximum(inp,  0)

    def relu_prime(self, inp):
        inp[inp <= 0] = 0
        inp[inp > 0] = 1
        return inp

    def softmax(self, inp):
        return (np.e ** inp) / (sum(np.e ** inp))

    def softmax_prime(self, inp):
        pass

    def sigmoid(self, inp):
        return 1.0/(1.0+np.exp(-inp))

    def sigmoid_prime(self, inp):
        return self.sigmoid(inp) * (1 - self.sigmoid(inp))


def scale(X, x_min=0, x_max=1):
    nom = (X-X.min())*(x_max-x_min)
    denom = X.max() - X.min()
    denom = denom + (denom is 0)
    return x_min + nom/denom


df = pd.read_csv("Data/mnist_train.csv")
y = df.label
x = df.drop("label", axis=1)
x = x.values.reshape(60000, 28, 28)
x = scale(x)

enc = OneHotEncoder()
y = enc.fit_transform(y.values.reshape(-1, 1)).toarray()

temp = Dense([16], 10, 28, 28, ["relu", "sigmoid"])

temp.train(x, y, 500, 2)

# print(temp.)
