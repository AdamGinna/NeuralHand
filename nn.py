import numpy as np
import matplotlib.pyplot as plt

class Neural_Network(object):

  def __init__(self, eta=0.01):
    self.input_size = 2
    self.hidden_size = 10
    self.output_size = 2
    self.eta = eta
    self.errors = []

    self.W1 = np.random.randn(self.input_size, self.hidden_size)
    self.W2 = np.random.randn(self.hidden_size, self.output_size)

  def sigmoid(self, x):
    return 1./(1 + np.exp(-x))

  def sigmoid_derivative(self, s):
    return s * (1 - s) 

  def forward(self, x):
    self.y0 = np.array(x).copy()
    self.a1 = np.dot(self.y0, self.W1)
    self.y1 = self.sigmoid(self.a1)
    self.a2 = np.dot(self.y1, self.W2)
    self.y2 = self.sigmoid(self.a2)
    return self.y2

  def backward(self, output):
    self.epsilon_2 = output - self.y2
    self.delta_2 = self.epsilon_2 * self.sigmoid_derivative(self.y2)

    self.epsilon_1 = self.delta_2.dot(self.W2.T)
    self.delta_1 = self.epsilon_1 * self.sigmoid_derivative(self.y1)
    
    self.W2 += self.eta * self.y1.T.dot(self.delta_2)
    self.W1 += self.eta * self.y0.T.dot(self.delta_1)

  def train(self, x, y):
    self.forward(x)
    self.backward(y)
    self.errors.append(np.mean(np.square(y - self.y2)))