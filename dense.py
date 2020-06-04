import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class Dense:
    def __init__(self, h_layers, n_outputs, img_rows, img_cols, act_funcs):

        # TODO clean up the init method

        self.img_size = img_rows * img_cols

        # n_outputs is appended this way for clarity of user
        self.h_layers = h_layers.append(n_outputs)

        self.n_layers = len(h_layers)

        # train_error with track the average loss after each mini-batch
        self.train_error = []

        # Create list of activation funcs and list of derivatives of activation funcs
        act_func_dict = {"relu": self.relu,
                         "sigmoid": self.sigmoid, "softmax": self.softmax}
        if act_funcs:
            self.act_funcs = []
            for func in act_funcs:
                self.act_funcs.append(act_func_dict[func])
        else:
            self.act_funcs = [self.sigmoid for _ in range(self.n_layers)]
        act_func_prime_dict = {self.relu: self.relu_prime,
                               self.sigmoid: self.sigmoid_prime, self.softmax: self.softmax_prime}
        self.act_funcs_prime = []
        for func in self.act_funcs:
            self.act_funcs_prime.append(act_func_prime_dict[func])
        assert len(self.act_funcs) == self.n_layers

        # Initialize biases to random initialization in shape of hidden layers
        self.biases = np.array([np.random.randn(y) for y in h_layers])

        # initialize weights with std. deviation that divides the random initializations
        # by the sqrt of the number of inputs / 2
        # see cs231n Andrej Karpathy time-stamp 49:00 https://youtu.be/gYpoJMlgyXA?t=2942
        self.weights = [np.random.randn(
            self.img_size, h_layers[0]) / np.sqrt(self.img_size/2)]
        for i in range(self.n_layers-1):
            self.weights.append(np.random.randn(
                h_layers[i], h_layers[i+1]) / np.sqrt(h_layers[i] / 2))
        self.weights = np.array(self.weights)

    def forward(self, inp, train=False):
        """Passes flattened data through a the weights and biases and returns the output. 
        When train=True it also returns the activations and z_values for each layer"""

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

    def train(self, data, labels, epochs=500, batch_size=10, lr=.22):
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

                # TODO Change the loss function (really only affects the outcome in the computed gradient)
                loss = -np.sum(label * np.log(a[-1]))
                #loss = (a[-1] - label)
                self.batch_error.append(loss.mean())

                # set the final activ. grad. to deriv of loss function evaluated
                # with activation of final layer == result of network
                d_a = a[-1] - label

                # derivaties for current z_nodes
                d_z = 0
                for i in range(1, len(z)+1):

                    # evaluate the derivative of z_nodes using the derivative of the activ. func used,
                    # evaluated at zL, and mult. by the leading grad, because of chain rule,
                    # that is stored at d_a[i]
                    d_z = self.act_funcs_prime[-i](z[-i]) * d_a

                    # The grad. of biases is just that of z_nodes
                    img_d_b.append(d_z)

                    # compute the grads for each weight by multiplying the previous grads
                    # (d_z) by the actvs. in previous layer
                    # This creates multiple copies of the activations list so that each
                    # can be multiplied element-wise by the prev grads
                    layer_d = np.array([a[-i-1]] * len(d_z)).T * d_z
                    img_d_w.append(layer_d)
                    d_a = np.dot(d_z, self.weights[-i].T)

                batch_d_b.append(img_d_b)
                batch_d_w.append(np.array(img_d_w))

            # Compute the average gradient across the mini-batch for weights and biases
            # and mult. by learning rate
            batch_d_w = np.array(batch_d_w).mean(axis=0) * lr
            batch_d_b = np.array(batch_d_b).mean(axis=0) * lr

            # Update the network weights and biases by subtracting the gradients, which
            # are stored in reverse order
            self.weights = self.weights - batch_d_w[::-1]
            self.biases = self.biases - batch_d_b[::-1]

            # Log the average error from that batch
            self.train_error.append(np.array(self.batch_error).mean())

    def _get_batch(self, batch_size):
        """Internal function to get a mini-batch"""
        indices = np.random.randint(len(self.data), size=batch_size)
        # indices = [3305]  # Used occasionaly to ensure the network can overfit
        return self.data[indices].reshape(batch_size, 784), self.labels[indices]

    # Define the activation functions and their derivatives

    def relu(self, inp):
        return np.maximum(inp,  0)

    def relu_prime(self, inp):
        inp[inp <= 0] = 0
        inp[inp > 0] = 1
        return inp

    def softmax(self, inp):
        e_x = np.exp(inp - np.max(inp))
        return e_x / e_x.sum(axis=0)

    def softmax_prime(self, z):
        """Computes the gradient of the softmax function.
        z: (T, 1) array of input values where the gradient is computed. T is the
        number of output classes.
        Returns D (T, T) the Jacobian matrix of softmax(z) at the given z. D[i, j]
        is DjSi - the partial derivative of Si w.r.t. input j.
        """
        Sz = self.softmax(z)
        # -SjSi can be computed using an outer product between Sz and itself. Then
        # we add back Si for the i=j cases by adding a diagonal matrix with the
        # values of Si on its diagonal.
        D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
        return D

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

temp = Dense([10, 10], 10, 28, 28, ["relu", "relu", "softmax"])

temp.train(x, y, 500, 30, .1)

# print(temp.)
