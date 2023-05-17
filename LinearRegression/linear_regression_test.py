import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go

from LinearRegression.linear_regression import LinearRegression

data = pd.read_csv('../data/world-happiness-report-2017.csv')
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

input_para = 'Economy..GDP.per.Capita.'
input_para2 = 'Freedom'
output_para = 'Happiness.Score'

x_train = train_data[[input_para, input_para2]].values
y_train = train_data[[output_para]].values

x_test = test_data[[input_para, input_para2]].values
y_test = test_data[[output_para]].values

# Configure the plot with training dataset.
plot_training_trace = go.Scatter3d(
    x=x_train[:, 0].flatten(),
    y=x_train[:, 1].flatten(),
    z=y_train.flatten(),
    name='Training Set',
    mode='markers',
    marker={
        'size': 10,
        'opacity': 1,
        'line': {
            'color': 'rgb(255, 255, 255)',
            'width': 1
        },
    }
)


plot_test_trace = go.Scatter3d(
    x=x_test[:, 0].flatten(),
    y=x_test[:, 1].flatten(),
    z=y_test.flatten(),
    name='Test Set',
    mode='markers',
    marker={
        'size': 10,
        'opacity': 1,
        'line': {
            'color': 'rgb(255, 255, 255)',
            'width': 1
        },
    }
)


plot_layout = go.Layout(
    title='Date Sets',
    scene={
        'xaxis': {'title': input_para},
        'yaxis': {'title': input_para2},
        'zaxis': {'title': output_para}
    },
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
)

plot_data = [plot_training_trace, plot_test_trace]

plot_figure = go.Figure(data=plot_data, layout=plot_layout)

plotly.offline.plot(plot_figure)

num_iter = 500
learning_rate = 0.01
LR = LinearRegression(x_train, y_train)
(theta, cost) = LR.train(learning_rate, num_iter)

print('Loss at Begin', cost[0])
print('Loss at End', cost[-1])

plt.plot(range(num_iter), cost)
plt.xlabel('Iter')
plt.ylabel('Loss')
plt.title('Gradient')
plt.show()

predictions_num = 10
x_min = x_train[:, 0].min()
x_max = x_train[:, 0].max()

y_min = x_train[:, 1].min()
y_max = x_train[:, 1].max()


x_axis = np.linspace(x_min, x_max, predictions_num)
y_axis = np.linspace(y_min, y_max, predictions_num)


x_predictions = np.zeros((predictions_num * predictions_num, 1))
y_predictions = np.zeros((predictions_num * predictions_num, 1))

x_y_index = 0
for x_index, x_value in enumerate(x_axis):
    for y_index, y_value in enumerate(y_axis):
        x_predictions[x_y_index] = x_value
        y_predictions[x_y_index] = y_value
        x_y_index += 1

z_predictions = LR.predict(np.hstack((x_predictions, y_predictions)))

plot_predictions_trace = go.Scatter3d(
    x=x_predictions.flatten(),
    y=y_predictions.flatten(),
    z=z_predictions.flatten(),
    name='Prediction Plane',
    mode='markers',
    marker={
        'size': 1,
    },
    opacity=0.8,
    surfaceaxis=2,
)

plot_data = [plot_training_trace, plot_test_trace, plot_predictions_trace]
plot_figure = go.Figure(data=plot_data, layout=plot_layout)
plotly.offline.plot(plot_figure)
