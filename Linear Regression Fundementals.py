import matplotlib.pyplot as plt
import csv
from pandas import *

data = read_csv("features.csv")
 
# converting column data to list
step = data['Date'].tolist()

months = [i for i in range(len(step))]
revenue = data['Unemployment'].tolist()


 
def get_gradient_at_b(x, y, b, m):
  N = len(x)
  diff = 0
  for i in range(N):
    x_val = x[i]
    y_val = y[i]
    diff += (y_val - ((m * x_val) + b))
  b_gradient = -(2/N) * diff  
  return b_gradient

def get_gradient_at_m(x, y, b, m):
  N = len(x)
  diff = 0
  for i in range(N):
      x_val = x[i]
      y_val = y[i]
      diff += x_val * (y_val - ((m * x_val) + b))
  m_gradient = -(2/N) * diff  
  return m_gradient

#step_gradient function here
def step_gradient(b_current, m_current, x, y, learning_rate):
    b_gradient = get_gradient_at_b(x, y, b_current, m_current)
    m_gradient = get_gradient_at_m(x, y, b_current, m_current)
    b = b_current - (learning_rate * b_gradient)
    m = m_current - (learning_rate * m_gradient)
    return [b, m]
  
#gradient_descent function here:  
def gradient_descent(x, y, learning_rate, num_iteration):
  b , m = 0, 0
  for i in range(num_iteration):
    b, m = step_gradient(b, m, x, y, learning_rate)
  return [b, m]



b, m = gradient_descent(months, revenue, 0.01, 1000)

y = [m*x + b for x in months]

plt.plot(months, revenue, "o")
plt.plot(months, y)

plt.show()