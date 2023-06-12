# Program that implements the K-Means Clustering algorithm
# and a neural network method on the Iris dataset
"""
This program implements the K-Means Clustering algorithm on the Iris dataset
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from k_means import KMeans
from neural_network import NeuralNetwork

DO_K_MEANS = False
DO_NEURAL_NETWORK = True

if DO_K_MEANS:
    # Importing the dataset
    df = pd.read_csv('iris.csv', names=[
        'Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class'])

    # Distribution plot of the Sepal Length by Class using displot
    for i in range(4):
        plt.figure()
        sns.displot(df, x=df.columns[i], hue='Class', kind='kde')
        plt.savefig(f'figs/{df.columns[i]}.png')

    # Applying k-means to the dataset / Creating the k-means classifier
    kmeans2 = KMeans(2)
    result = kmeans2.fit(df)
    kmeans2.make_final_png(df, result[0], result[1], result[2])
    kmeans2.make_obj_png(result)

    kmeans3 = KMeans(3)
    result = kmeans3.fit(df)
    kmeans3.make_final_png(df, result[0], result[1], result[2])
    kmeans3.make_obj_png(result)
    kmeans3.make_decision_boundary_png(df, result[0], result[1], result[2])

if DO_NEURAL_NETWORK:
    # Applying neural network to the dataset
    df = pd.read_csv('iris.csv', names=[
        'Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class'])
    data = df.iloc[:, [0, 1, 2, 3]].values
    nn = NeuralNetwork()

    types = []
    for type in df['Class'].values:
        if type == 'Iris-setosa':
            types.append(None)
        elif type == 'Iris-versicolor':
            types.append(0.0)
        elif type == 'Iris-virginica':
            types.append(1.0)
    df['Class'] = types

    # IGNORE THIS
    # weights = [-0.71, -0.02244, -0.07851, 0.13069, 0.25582]
    # weights = [-0.79, -0.02859, -0.0857, 0.15407, 0.27964]
    # weights = [-0.94, -0.03696, -0.0916, 0.18685, 0.30768]
    # weights = [-0.74, -0.02219, -0.07995, 0.13523, 0.26154]
    # # .02
    # weights = [-0.43, -0.00356, -0.054582, 0.066802, 0.166809]
    # weights = [-0.151, 0.001285, -0.026252, 0.020975, 0.068346]
    # weights = [-0.101, -0.004613, -0.009198, 0.020745, 0.032075]
    # .02

    # This is the weights with large error.
    weights_bad = [-1, -0.06, -0.1, 0.2, 0.3]
    # These are the weights with very small error.
    weights = [-0.079, -0.00257, -0.008416, 0.01485, 0.027532]

    print(nn.mse(data[50:],
                 weights,
                 df['Class'].values[50:]))
    print(nn.mse(data[50:],
                 weights_bad,
                 df['Class'].values[50:]))
    nn.plot_part_b(df[50:], data[50:], weights)
    nn.plot_part_d(df[50:], data[50:], weights_bad, 'bad')
    nn.plot_part_d(df[50:], data[50:], weights, 'good')
