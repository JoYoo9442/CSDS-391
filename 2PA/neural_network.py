"""
Program that implements a neural network on the Iris dataset
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class NeuralNetwork:
    """Implementation of a neural network"""

    def __init__(self):
        pass

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def calculate_output(self, inputs, weights):
        """Calculate the output given the input and weights"""
        output = []

        for i, point in enumerate(inputs):
            # Regarding the bias term, we add a 1 to the input
            term = np.array([1])
            term = np.concatenate((term, point), axis=0)

            x = np.dot(term, weights)
            x = self.sigmoid(x)

            # Make sure the output is 0 or 1
            if x < 0.5:
                x = 0
            else:
                x = 1

            output.append(x)

        return np.array(output)

    def mse(self, data, weights, target):
        """Mean Squared Error"""
        # Calculate the output
        output = self.calculate_output(data, weights)
        # Calculate the mean squared error
        return np.mean((output - target)**2)

    def sum_gradient(self, data, weights, target):
        """Calculate the sum of the gradients"""
        # Initialize the gradients list
        gradients = []

        # Calculate the output
        y = self.calculate_output(data, weights)

        # Calculate the bias gradient
        bias = 0
        for i in range(len(data)):
            bias += (y[i] - target[i])

        gradients.append(bias)

        # Calculate the gradient for each weight
        for i in range(1, len(weights)):
            gradient = 0

            # Calculate the gradient for each data point
            for j, point in enumerate(data):
                gradient += (y[j] - target[j]) * point[i-1]

            # Append the gradient to the list
            gradients.append(gradient/len(data))

        return np.array(gradients)

    def train(self, data, target, weights, learning_rate=0.1, max_iter=1000):
        """Train the neural network"""
        for i in range(max_iter):
            gradients = self.sum_gradient(data, weights, target)
            weights -= learning_rate * gradients

            mse = self.mse(data, weights, target)
            if (mse == 0.0):
                break

        return weights

    def predict(self, data, weights):
        """Predict the class of a data point"""
        predictions = []
        outputs = self.calculate_output(data, weights)

        for i, point in enumerate(data):
            output = outputs[i]
            predictions.append(point.tolist() + [output])

        return np.array(predictions)

    def plot_part_b(self, df, data, weights):
        """Plot the data points and the decision boundary"""
        # Plot the data points
        plt.figure()
        sns.scatterplot(x=df['Sepal Length'], y=df['Petal Length'],
                        hue=self.calculate_output(data, weights),
                        palette=['red', 'blue'])
        plt.xlabel('Sepal Length')
        plt.ylabel('Petal Length')
        plt.title('Sepal Length vs. Petal Length')

        # Plot the decision boundary
        plt.plot((4.8, 8), (4.4, 5.2))

        plt.savefig('figs/part_b.png')

    def plot_part_d(self, df, data, weights, name):
        """Plot the data points and the decision boundary"""
        # Plot the data points
        plt.figure()
        sns.scatterplot(x=df['Sepal Width'], y=df['Petal Width'],
                        hue=self.calculate_output(data, weights),
                        palette=['red', 'blue'])
        plt.xlabel('Sepal Width')
        plt.ylabel('Petal Width')
        plt.title('Sepal Width vs. Petal Width')

        # Plot the decision boundary
        # plt.plot((2.5, 4), (0.5, 1.5))

        plt.savefig(f'figs/part_d{name}.png')
