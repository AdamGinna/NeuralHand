import numpy as np
import matplotlib.pyplot as plt

class Examples(object):

  def __init__(self, arm_length=10):
    self.arm_length = arm_length
    self.center = Point(0, 0)
    self.input = []
    self.output = []

  def generate(self, number_of_examples):
    for i in range(number_of_examples):
      point = self.generate_point()
      self.input.append([point.x, point.y])
    return self.input, self.output

  def generate_point(self):
    alpha = np.random.random() * np.pi
    beta = np.random.random() * np.pi
    self.output.append([alpha, beta])
    temppoint = self.translate(self.center, alpha)
    finalpoint = self.translate(temppoint, np.pi - beta + alpha )
    return finalpoint

  def translate(self, center, angle):
    return Point(center.x + self.arm_length * np.sin(angle), center.y - self.arm_length * np.cos(angle))

class Point(object):

  def __init__(self, x, y):
    self.x = x
    self.y = y