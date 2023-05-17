import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go

from LinearRegression.linear_regression import LinearRegression

data = pd.read_csv('../data/non-linear-regression-x-y.csv')
x = data['x'].values.reshape((data.shape[0], 1))
y = data['y'].values.reshape((data.shape[0], 1))

data.head(10)

plt.plot(x, y)
plt.show()

num_iter = 50000
learning_rate = 0.01
polynomial_degree = 15
sinusoid_degree = 15
normalize = True
LR = LinearRegression(x, y, polynomial_degree, sinusoid_degree, normalize)
(theta, cost) = LR.train(learning_rate, num_iter)

print('Loss at Begin', cost[0])
print('Loss at End', cost[-1])

plt.plot(range(num_iter), cost)
plt.xlabel('Iter')
plt.ylabel('Loss')
plt.title('Gradient')
plt.show()

