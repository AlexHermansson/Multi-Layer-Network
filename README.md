# Multi-Layer-Network
A short implementation of a multi-layered neural network.

Technical details:
The cross-entropy loss is used together with ReLu activation function. There is also support for computing numerical gradients in order to make sure that the implementation of the analytical gradients is correct. Optimization is done using stochastic gradient descent with momentum, or using Adam optimizer, see https://arxiv.org/pdf/1412.6980.pdf.

The test cases is with the CIFAR-10 data set, which can be 
downloaded from https://www.cs.toronto.edu/~kriz/cifar.html.

