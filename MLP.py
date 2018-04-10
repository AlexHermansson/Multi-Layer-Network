import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming


class Two_layer_network():
    """ A two layer neural network class"""

    def __init__(self, K, d, m, He = False):
        """
        Initialization of the network
        :param K: Number of classes. int
        :param d: Number of dimensions of input data. int
        :param m: Number of hidden nodes. int
        :param He: If He initialization of weights
        """

        # He initialization of weights
        if He:
            std = np.sqrt(2/d)
        else:
            std = 0.001
        self.m = m
        self.leaky_relu = leaky_relu
        self.W1 = np.random.normal(0, std, (m, d))  # m x d
        self.W2 = np.random.normal(0, std, (K, m))  # K x m
        self.b1 = np.zeros((m, 1))                  # m x 1
        self.b2 = np.zeros((K, 1))                  # K x 1

    def evaluate_classifier(self, X):
        """
        A forward pass through the network. Samples should be put columnwise in data matrix.
        :param X: Data batch. d x n
        :return: Probabilities for each class. K x n
        """

        S1 = np.dot(self.W1, X) + self.b1
        H = np.maximum(0, S1) # ReLu activation element wise
        S = np.dot(self.W2, H) + self.b2
        P = self.softmax(S)

        return P, S1, H

    def softmax(self, S):
        """
        Softmax applied element wise to a score matrix
        :param S: Scores. K x n
        :return: Probabilities. K x n
        """

        # To ensure no overflow happens
        S -= np.max(S, axis=0)
        return np.exp(S)/np.sum(np.exp(S), axis=0)

    def predict(self, X):
        """
        Predicts the classes of data X.
        :param X: Data batch. d x n
        :return: Output class prediction. A vector of n values in range [0, K]
        """

        P, _, _ = self.evaluate_classifier(X)
        return np.argmax(P, axis=0)

    def compute_cost(self, X, Y, lambd):
        """
        Cost function
        :param X: Data batch. d x n
        :param Y: Labels, one hot encoded. K x n
        :param lambd: Regularization parameter
        :return: Cost, a scalar value
        """

        n = X.shape[1] # number of samples in batch
        P, _, _ = self.evaluate_classifier(X)  # K x n

        # take only the diagonal of the matrix multiplication Y.T @ P, i.e. all the dot products dot(y.T, p)
        diag = np.einsum('ij, ji -> i', Y.T, P) # vector of n values
        L = -np.sum(np.log(diag))/n

        R = np.sum(self.W1**2) + np.sum(self.W2**2)

        return L + lambd*R

    def compute_accuracy(self, X, y):
        """
        Computes accuracy in percent. Uses scipy lib.
        :param X: Data batch. d x n
        :param y: Labels batch. n vector
        :return: Prediction accuracy in percent
        """

        return 1 - hamming(y, self.predict(X))

    def compute_gradients(self, X, Y, lambd):
        """
        A function that evaluates, for a mini-batch, the gradients of the
        cost function w.r.t W and b.
        :param X: Data batch. d x n
        :param Y: Labels batch. K x n
        :param lambd: Regularization parameter. lambd >= 0
        :return: Gradients: grad_W1, grad_b1, grad_W2, grad_b2
        """

        n = X.shape[1]
        P, S1, H = self.evaluate_classifier(X) # K x n, m x n, and m x n
        G = P - Y # K x n

        grad_b2 = 1/n * np.sum(G, axis=1).reshape(-1, 1)
        grad_W2 = 1/n * np.dot(G, H.T) + 2*lambd*self.W2

        G = np.dot(self.W2.T, G) # m x n
        temp = np.where(S1 > 0, 1, 0)
        # Instead of making diagonal matrices for every example, do element wise matrix multiplication.
        G = G*temp

        grad_b1 = 1/n * np.sum(G, axis=1).reshape(-1, 1)
        grad_W1 = 1/n * np.dot(G, X.T) + 2*lambd*self.W1

        return grad_W1, grad_b1, grad_W2, grad_b2

    def compute_num_grads_center(self, X, Y, lambd, h = 1e-5):
        """
        A somewhat slow method to numerically approximate the gradients using the central difference.
        Used to be able to check that the analytical gradients are correct.
        :param X: Data batch. d x n
        :param Y: Labels batch. K x n
        :param lambd: Regularization parameter
        :param h: Step length, default to 1e-5. Should obviously be kept small.
        :return: Approximate gradients grad_W, K x d and grad_b, K x 1
        """

        # df/dx ≈ (f(x + h) - f(x - h))/2h according to the central difference formula

        params = [self.W1, self.b1, self.W2, self.b2]
        num_grads = []

        # Iterate over all parameters in the model
        for param in params:

            grad = np.zeros(param.shape)

            # Iterate over all dimensions of the parameter
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:

                ix = it.multi_index
                old_value = param[ix]
                param[ix] = old_value + h
                plus_cost = self.compute_cost(X, Y, lambd)
                param[ix] = old_value - h
                minus_cost = self.compute_cost(X, Y, lambd)
                param[ix] = old_value # Restore original value

                grad[ix] = (plus_cost - minus_cost) / (2*h)
                it.iternext() # go to next index

            num_grads.append(grad)

        return num_grads

    def train(self, X, Y, epochs, eta, rho, batch_size = 100, lambd = 0,
              decay = 1,X_val = np.array([None]), Y_val = np.array([None]),
              stop_criteria = 10, Adam = True):
        """
        Training the network. Uses early stopping if a validation set is 
        available to avoid overfitting.
        :param X: Data. d x N
        :param Y: Labels. K x N
        :param epochs: Number of epochs
        :param eta: Learning rate
        :param rho: Momentum parameter
        :param batch_size: Batch size, default to 100
        :param lambd: L2 Regularization parameter, default to zero which means no regularization
        :param decay: Learning rate decay parameter. 1 is default and is equivalent to no decay.
        :param X_val: Validation data. d x n
        :param Y_val: Validation labels. K x n
        :param stop_criteria: If no improvement on validation loss, stop early and save best parameters.
        :param Adam. boolean, if using Adam optimizer instead of SGD with momentum.
        :return: A trained version of the network itself and lists of cost for the different epochs.
        """

        assert type(batch_size) == int
        assert type(epochs) == int

        N = X.shape[1]

        # Boolean to keep track of if a validation set is used
        validation_set = bool(X_val.any()) and bool(Y_val.any())

        C_val = []
        C_train = []
        best_val_cost = np.inf
        count = 0

        # Just to make it easier to make updates of the parameters
        params = [self.W1, self.b1, self.W2, self.b2]
        V = [np.zeros(param.shape) for param in params]

        # Initialize Adam optimizer parameters
        if Adam:
            beta_1 = 0.9; beta_2 = 0.999; eps = 1e-8
            M = [np.zeros(param.shape) for param in params]
            t = 0

        for epoch in range(epochs):

            # To keep track on the epoch number while training
            if epoch % 5 == 0:
                print('epoch: {}'.format(epoch))

            for batch_index in range(N//batch_size):

                # To get indices of batch
                start_index = int(batch_index * batch_size)
                last_index = start_index + batch_size
                # special case if N is not evenly divisible by batch_size
                if last_index > N - 1:
                    last_index = N - 1


                # Adam optimizer, see: https://arxiv.org/pdf/1412.6980.pdf
                if Adam:

                    t = t+1
                    grads = self.compute_gradients(X[:, start_index:last_index], Y[:, start_index:last_index], lambd)

                    # Calculate first and second order moments
                    M = [beta_1*m + (1-beta_1)*g for m, g in zip(M, grads)]
                    V = [beta_2*v + (1-beta_2)*(g**2) for v, g in zip(V, grads)]
                    M_hat = [m/(1-beta_1**t) for m in M]
                    V_hat = [v/(1-beta_2**t) for v in V]

                    # Update parameters
                    params = [param - eta*m_hat/(np.sqrt(v_hat) + eps) for param, v_hat, m_hat in zip(params, V_hat, M_hat)]
                    self.W1, self.b1, self.W2, self.b2 = params

                # Standard SGD with momentum
                else:

                    grads = self.compute_gradients(X[:, start_index:last_index], Y[:, start_index:last_index], lambd)

                    # Calculate momentum terms
                    V = [rho*v + eta*g for v, g in zip(V, grads)]

                    # Update parameters
                    params = [param - v for param, v in zip(params, V)]
                    self.W1, self.b1, self.W2, self.b2 = params

            cost_train = self.compute_cost(X, Y, lambd)
            C_train.append(cost_train)

            if validation_set:
                cost_val = self.compute_cost(X_val, Y_val, lambd)
                C_val.append(cost_val)

                # If validation cost improves, save parameters
                if cost_val < best_val_cost:
                    count = 0
                    best_val_cost = cost_val
                    best_params = (self.W1, self.b1, self.W2, self.b2)

                # If not, increment counter and if counter is larger than stop critera, 
                # stop training prematurely.
                else:
                    count += 1
                    if count > stop_criteria:
                        print('Stopping early.')
                        break

            # decay the learning rate
            eta *= decay

        # otherwise, we don't have any best params at all..
        if validation_set:
            self.W1, self.b1, self.W2, self.b2 = best_params

        return C_train, C_val

# Some help functions for loading data and for doing a hyper parameter search

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')

    return dict

def to_one_hot(labels):
    """
    Converts labels to a one hot encoding.
    :param labels: A list of labels for the training data. N samples, and K different classes.
    :return: the one hot representation of the labels on form K x N
    """
    N = len(labels)
    K = len(np.unique(labels)) # number of classes
    Y = np.zeros((K, N), dtype=int)
    Y[labels, np.arange(N)] = 1

    return Y

def load_batch(filename):
    data_dict = unpickle(filename)
    X = data_dict[b'data']
    y = np.array(data_dict[b'labels'])
    X = (X / 255).T  # rescale to be in [0, 1] and transpose to be on form d x N
    Y = to_one_hot(y)

    return X, Y, y

def zero_mean_transf(X_train, X_val, X_test):
    """Transform X_train to have zero mean, then remove same mean
    from X_val and X_test."""
    X_mean = np.mean(X_train, axis=1).reshape(-1, 1)

    return X_train - X_mean, X_val - X_mean, X_test - X_mean

def cost_plot(C_train, C_val, acc_test=0):
    # Plotting cost vs epochs
    e = np.arange(len(C_train))
    plt.plot(e, C_train, label = 'Training loss')
    plt.plot(e, C_val, label = 'Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('$\lambda$ = %s, $\eta$ = %s, test acc. = %s, decay = %s, $rho$ = %s' % (lambd, lr, round(acc_test*100, 2), decay, rho))
    #plt.savefig('fig')
    plt.show()


def hyper_param_search(iterations=50):
    # fix parameters
    K, d, m = 10, 3072, 50
    epochs = 50; rho = 0.9
    batch_size = 100; decay = 1

    e_min = -3.8; e_max = -2.8
    l_min = -4; l_max = -3
    for i in range(iterations):
        # sample eta and lambda
        e = e_min + (e_max - e_min)*np.random.uniform(0, 1)
        l = l_min + (l_max - l_min)*np.random.uniform(0, 1)
        eta = 10 ** e
        lambd = 10**l

        net = Two_layer_network(K, d, m, True)
        net.train(X_train, Y_train, epochs, eta, rho,
                  batch_size, lambd,decay, X_val, Y_val)
        acc_v = net.compute_accuracy(X_val, y_val)
        #print('iteration ', (i+1))
        print('{}% accuracy. eta = {}, lambda = {}'.format(round(acc_v*100, 2), round(eta, 5), round(lambd, 5)))
        print()


#np.random.seed(0)

# Load batch 1 as training set, and batch 2 as validation set
X_train, Y_train, y_train = load_batch('data_batch_1')
X_val, Y_val, y_val = load_batch('data_batch_2')
X_test, Y_test, y_test = load_batch('test_batch')

# Transform data to have zero mean (only X_train have exactly zero mean)
X_train, X_val, X_test = zero_mean_transf(X_train, X_val, X_test)

# To do a search over eta and lambda
#hyper_param_search(50)

# Initialize the network
d, N = X_train.shape
K = Y_train.shape[0]
m = 50 # hidden nodes
He = True # If He init

network = Two_layer_network(K, d, m, He)


# Learning parameters
epochs = 50
stop_criteria = 10 # if validation error doesn't reduce in #stop_criteria epochs, stop training. Early stopping.
lr = 0.00025#0.0257
rho = 0.89
batch_size = 100
lambd = 0.001#0.0008#0.00194
decay = 1
Adam = True # If using the Adam optimizer instead of using stochastic SGD with momentum

# Train the network
C_train, C_val = network.train(X_train, Y_train, epochs, lr, rho, batch_size,
                               lambd, decay, X_val, Y_val, stop_criteria, Adam)

# Validation accuracy
acc_val = network.compute_accuracy(X_val, y_val)
print('{}% accuracy for the validation set'.format(round(acc_val*100, 2)))

# Test accuracy
acc_test = network.compute_accuracy(X_test, y_test)
print('{}% accuracy for the test set'.format(round(acc_test*100, 2)))

cost_plot(C_train, C_val, acc_test)