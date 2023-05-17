import numpy as np
from utils.features import prepare_for_training


class LinearRegression:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        Prepare data
        Get nums of features
        Initialize theta matrix
        """
        (data_processed,
         features_mean,
         features_derivation) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data)
        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_derivation = features_derivation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features, 1))

    def train(self, alpha, num_itera=500):
        cost_history = self.gradient_descent(alpha, num_itera)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_itera):
        cost_history = []
        for _ in range(num_itera):
            self.step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history

    def step(self, alpha):
        batch = self.data.shape[0]
        prediction = LinearRegression.hypothesis(self.data, self.theta)
        delta = prediction - self.labels
        theta = self.theta
        theta = theta - alpha * (1 / batch) * (np.dot(delta.T, self.data)).T
        self.theta = theta

    def cost_function(self, data, labels):
        prediction = LinearRegression.hypothesis(data, self.theta)
        delta = prediction - labels
        cost = (1 / 2) * np.dot(delta.T, delta) / data.shape[0]
        return cost[0][0]

    @staticmethod
    def hypothesis(data, theta):
        prediction = np.dot(data, theta)
        return prediction

    def get_cost(self, data, labels):
        data_pre = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree,
                                        self.normalize_data)[0]
        return self.cost_function(data_pre, labels)

    def predict(self, data):
        data_pre = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree,
                                        self.normalize_data)[0]
        predictions = LinearRegression.hypothesis(data_pre, self.theta)
        return predictions
