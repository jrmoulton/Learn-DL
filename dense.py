import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class Dense:
    def __init__(self, h_layers, img_rows, img_cols, act_funcs):

        # TODO clean up the init method
 
        self.img_size = img_rows * img_cols
        self.n_layers = len(h_layers)
        self.h_layers = h_layers

        # train_error with track the average loss after each mini-batch
        self.train_error = []

        # Create list of activation funcs and list of derivatives of activation funcs
        act_func_dict = {"relu": self.relu, "sigmoid": self.sigmoid,
                         "softmax": self.softmax, "mrelu": self.mrelu}
        self.act_funcs = []
        for func in act_funcs:
            self.act_funcs.append(act_func_dict[func])

        act_func_prime_dict = {self.relu: self.relu_prime,
                               self.sigmoid: self.sigmoid_prime,
                               self.softmax: self.softmax_prime,
                               self.mrelu: self.mrelu_prime}
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

        # Initialize paramaters for momentum
        self.v_w = np.zeros(self.weights.shape)
        self.v_b = np.zeros(self.biases.shape)

        # End of init method

    def forward(self, inp, train=False):
        """Passes flattened data through a the weights and biases and returns the output.
        When train=True it also returns the activations and z_values for each layer"""

        zipped_b_w = list(zip(self.biases, self.weights))
        if train:
            z = []
            a = [inp]
            for i in range(self.n_layers):
                b, w = zipped_b_w[i]
                inp = np.dot(inp, w) + b
                z.append(inp)
                inp = self.act_funcs[i](inp)
                a.append(inp)
            return a, z
        else:
            for i in range(self.n_layers):
                b, w = zipped_b_w[i]
                inp = self.act_funcs[i](np.dot(inp, w) + b)
            return inp

    # End of forward pass

    def train(self, data, labels, epochs=500, batch_size=10, lr=.22):
        self.data = data
        self.labels = labels

        # Loop to repeat the mini-batch training process a defined number of steps/epochs
        for _ in range(epochs):
            # Print out training loss and validation accuracy every 100 epochs
            if _ % 101 == 0:
                print(
                    f"train error is: {np.array(self.train_error[-10:]).mean()}")
                print(f"Acc is: {test_func(temp, x_test, y_test, 2000)}")
                print()

            x_batch, y_batch = self._get_batch(batch_size)

            # Lists to store the gradients for each image for weights and biases
            batch_d_b = []
            batch_d_w = []

            # List to store the loss for each image that will be averaged and added to self.train_error
            self.batch_error = []

            # Loop for each image to compute gradients and store them to averaged later
            for img, label in zip(x_batch, y_batch):
                a, z = self.forward(img, train=True)
                img_d_b = []
                img_d_w = []

                # Cross entropy loss function
                # loss = -np.sum(label * np.log(a[-1]))

                # MSE loss function
                loss = (a[-1] - label) ** 2
                self.batch_error.append(loss.mean())

                # set the final activ. grad. to deriv of loss function evaluated
                # with activation of final layer/result of network
                d_a = (a[-1] - label) * 2

                # Do the backward pass to compute gradients still on one image for each weight and bias
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

            # Compute and update momentum for the network (not necessary, just an improvement)
            # Momentum seems to not permorm as well on networks with fewer layers. More layers is
            # where momentum shines. Andrew Ng deepleaning.ai
            beta = .9
            self.v_w = (beta * self.v_w) + ((1-beta) * batch_d_w[::-1])
            self.v_b = (beta * self.v_b) + ((1-beta) * batch_d_b[::-1])

            # Update the network weights and biases by subtracting the computed momentum * learning rate
            self.weights = self.weights - self.v_w
            self.biases = self.biases - self.v_b

            # Log the average error from that batch
            self.train_error.append(np.array(self.batch_error).mean())

        # End of training function

    def _get_batch(self, batch_size):
        """Internal function to get a mini-batch"""
        indices = np.random.randint(len(self.data), size=batch_size)
        # indices = [3305]  # Used occasionaly to ensure the network can overfit
        return self.data[indices].reshape(batch_size, 784), self.labels[indices]

    # Define the activation functions and their derivatives

    def relu(self, inp):
        return np.maximum(0, inp)

    def relu_prime(self, inp):
        inp[inp <= 0] = 0
        inp[inp > 0] = 1
        return inp

    def mrelu(self, inp):
        inp[inp <= 0] = inp[inp <= 0]/10
        return inp

    def mrelu_prime(self, inp):
        inp[inp <= 0] = .1
        inp[inp > 0] = 1
        return inp

    def softmax(self, inp):
        inp -= np.max(inp)
        inp = inp / inp.sum(axis=0)
        return inp

    def softmax_prime(self, inp):
        return inp * (1 - inp)

    def sigmoid(self, inp):
        return 1.0/(1.0+np.exp(-inp))

    def sigmoid_prime(self, inp):
        return self.sigmoid(inp) * (1 - self.sigmoid(inp))


def scale(X, x_min=0, x_max=1):
    nom = (X-X.min())*(x_max-x_min)
    denom = X.max() - X.min()
    denom = denom + (denom is 0)
    return x_min + nom/denom


x = pd.read_csv("Data/x_train.csv", index_col=0)
x = x.values.reshape(-1, 28, 28)
y = pd.read_csv("Data/y_train.csv", index_col=0).values

x_test = pd.read_csv("Data/x_test.csv", index_col=0)
x_test = x_test.values.reshape(-1, 28, 28)
y_test = pd.read_csv("Data/y_test.csv", index_col=0).values


def test_func(net_obj, x_test, y_test, batch_size):

    net_obj.data = x_test
    net_obj.labels = y_test
    x_test_batch, y_test_batch = net_obj._get_batch(batch_size)
    acc = []
    for i in range(len(x_test_batch)):
        if net_obj.forward(x_test_batch[i]).argmax() == y_test_batch[i].argmax():
            acc.append(True)
        else:
            acc.append(False)
    net_obj.data = x
    net_obj.labels = y
    return np.count_nonzero(np.array(acc))/batch_size*100


if __name__ == "__main__":
    temp = Dense(h_layers=[25, 18, 15, 10], img_rows=28, img_cols=28, act_funcs=[
        "relu", "relu", "relu", "sigmoid", ])

    temp.train(x, y, 1000, 30,)

    print(test_func(temp, x_test, y_test, 5000))
