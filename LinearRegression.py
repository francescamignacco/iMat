import numpy as np


class NeuralNetwork(object):
    def __init__(self):
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def forward(self, x):
        self.z2 = np.dot(x, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def costfunction(self, x, y):
        self.yHat = self.forward(x)
        J = 0.5*np.sum((y-self.yHat)**2)
        return J

    def sigmoidPrime(self, z):
        return np.exp(-z)/((1+np.exp(-z))**2)

    def costFunctionPrime(self, x, y):
        self.yHat = self.forward(x)
        self.delta3 = (-(y-self.yHat)*self.sigmoidPrime(self.z3))
        self.djdw2 = np.dot(self.a2.T, self.delta3)

        self.delta2 = np.dot(self.delta3, np.transpose(self.W2)*self.sigmoidPrime(self.z2))
        self.djdw1 = np.dot(np.transpose(x), self.delta2)
        return self.djdw1, self.djdw2


X = np.random.rand(100,2)
y = np.apply_along_axis(lambda element: element[0] + element[1], axis=1, arr=X)

NN = NeuralNetwork()
maxIter = 100
iteration = 0
learningRate = 0.01

while iteration < maxIter:
    D1, D2 = NN.costFunctionPrime(X, y)
    NN.W1 = NN.W1 - learningRate*D1
    if iteration == 0:
        print(NN.W2.shape)
        print(D2.shape)
    NN.W2 = NN.W2 - learningRate*D2
    iteration = iteration + 1
    if iteration % 10 == 0:
        print(iteration)
        print(NN.costfunction(X, y))