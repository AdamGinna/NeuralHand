import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(object):

  def __init__(self, eta=0.01, layers=(2,10,2)):
    self.eta = eta
    self.errors = []
    self.layers =[ np.random.randn(layers[_], layers[_+1]) for _ in range(len(layers)-1) ]
    self.output_size = layers[-1]
    self.input_size = layers[0]

  def sigmoid(self, x):
    return 1./(1 + np.exp(-x))

  def sigmoid_derivative(self, s):
    return s * (1 - s) 

  def forward(self, x):
    self.y = []
    self.y.append( np.array(x).copy() ) 
    for i in range(len(self.layers)):
      a = np.dot(self.y[i], self.layers[i])
      self.y.append(self.sigmoid(a))
    return self.y[-1]



  def backward(self, output):
    delta = []
    epsilon = output - self.y[-1]
    delta.append( epsilon * self.sigmoid_derivative(self.y[-1]) ) 
    for i in range(len(self.layers)-1,0,-1):
      epsilon = delta[-1].dot(self.layers[i].T)
      delta.append( epsilon * self.sigmoid_derivative(self.y[i]) ) 
    delta.reverse()
    for i in range(len(self.layers)):
      self.layers[i] += self.eta * self.y[i].T.dot(delta[i])
      

  def train(self, x, y):
    self.forward(x)
    self.backward(y)
    self.errors.append(np.mean(np.square(y - self.y[-1])))