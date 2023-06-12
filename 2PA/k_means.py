"""
This module contains the KMeans class, which is used to perform k-means
"""
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import seaborn as sns


class KMeans:
    """Implementation of the k-means algorithm"""

    def __init__(self, k, max_iter=100, tol=pow(10, -3)):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol

    def initial_centroids(self, data):
        """Randomly initialize the centroids"""
        centroids = []

        for i in range(self.k):
            centroids.append(data[np.random.randint(0, len(data))])

        return centroids

    def assign_cluster(self, data, centroids):
        """Assign each point to the nearest cluster"""
        clusters = []

        for point in data:
            point_cluster = []

            for centroid in centroids:
                # Compute the Euclidean distance between
                # the point and the centroid
                point_cluster.append(
                        norm(np.array(point) - np.array(centroid)))

            clusters.append(np.argmin(point_cluster))

        return clusters

    def generate_centroids(self, data, clusters):
        """Compute the centroids"""
        new_centroids = []

        for i in range(self.k):
            pt_cluster = []
            for j, point in enumerate(data):
                if clusters[j] == i:
                    pt_cluster.append(point)
            mean_c = np.mean(pt_cluster, axis=0)
            new_centroids.append(mean_c)

        return new_centroids

    def fit(self, df):
        """Perform k-means clustering"""
        i = -1
        clusters = []
        data = df.iloc[:, [0, 1, 2, 3]].values
        objective_function_values = []

        # Randomly initialize the centroids
        centroids = self.initial_centroids(data)

        while i < self.max_iter:
            i += 1

            # Assign each point to the nearest cluster
            clusters = self.assign_cluster(data, centroids)

            if i == 0:
                self.make_initial_png(df, clusters, centroids, data)
            elif i == 2:
                self.make_intermediate_png(df, clusters, centroids, data)

            # Calculate the objective function and append it to the list
            objective_function_values.append(
                    self.calculate_objective_function(data, clusters, centroids
                                                      ))

            # Compute the centroids
            centroids = self.generate_centroids(data, clusters)

        return (clusters, centroids, data, objective_function_values)

    def calculate_objective_function(self, data, clusters, centroids):
        """Calculate the objective function D"""
        point_by_cluster = self.split_data_by_cluster(data, clusters)
        objective_function = 0

        for i in range(self.k):
            for point in point_by_cluster[i]:
                objective_function += norm(
                        np.array(point) - np.array(centroids[i]))**2

        return objective_function

    def make_obj_png(self, result):
        plt.figure()
        s = sns.scatterplot(
                x=range(20),
                y=result[3][:20])
        s.set(
                xlabel='Iteration',
                ylabel='Objective Function',
                xticks=range(20))
        plt.plot(result[3][:20])
        plt.savefig(f'figs/obj_func{self.k}.png')

    # Ok what u need to do is make different plots at different times
    # So like a plot for the randomized centroids,
    # a plot for the first iteration, a plot for the second iteration, etc.

    def split_data_by_cluster(self, data, clusters):
        """Split the data by cluster"""
        point_by_cluster = [[], [], []]

        for i, point in enumerate(data):
            point_by_cluster[clusters[i]].append(list(point))

        return point_by_cluster

    def make_initial_png(self, df, clusters, centroids, data):
        self.make_png(df, clusters, centroids, data, f'initial-{self.k}')

    def make_intermediate_png(self, df, clusters, centroids, data):
        self.make_png(df, clusters, centroids, data, f'intermediate-{self.k}')

    def make_final_png(self, df, clusters, centroids, data):
        self.make_png(df, clusters, centroids, data, f'final-{self.k}')

    def make_png(self, df, clusters, centroids, data, name):
        """Make a png of the clusters"""
        # Which columns to use in visualization
        x_column = 0
        y_column = 2

        point_by_cluster = self.split_data_by_cluster(data, clusters)

        centroids_x = [centroids[i][x_column] for i in range(len(centroids))]
        centroids_y = [centroids[i][y_column] for i in range(len(centroids))]

        # x_data = df[df.columns[x_column]]
        # y_data = df[df.columns[y_column]]
        # clusters = result[0]
        # plt.scatter(x_data, y_data, c=clusters, cmap='rainbow')

        # Visualizing the clusters
        plt.figure()

        for i in range(self.k):
            x_data = [x[x_column] for j, x in enumerate(point_by_cluster[i])]
            y_data = [y[y_column] for j, y in enumerate(point_by_cluster[i])]
            plt.scatter(x_data, y_data, label=f'Cluster {i+1}')

        plt.plot(
                centroids_x,
                centroids_y,
                c='white',
                marker='.',
                linewidth='0.01',
                markerfacecolor='black',
                markersize='22')
        plt.title("K-Means Clustering")
        plt.xlabel(df.columns[x_column])
        plt.ylabel(df.columns[y_column])
        plt.legend()

        plt.savefig(f'figs/{name}.png')

    def make_decision_boundary_png(self, df, clusters, centroids, data):
        """Make a png of the decision boundary"""
        # Which columns to use in visualization
        x_column = 0
        y_column = 2

        point_by_cluster = self.split_data_by_cluster(data, clusters)

        centroids_x = [centroids[i][x_column] for i in range(len(centroids))]
        centroids_y = [centroids[i][y_column] for i in range(len(centroids))]

        # Visualizing the clusters
        plt.figure()

        for i in range(self.k):
            x_data = [x[x_column] for j, x in enumerate(point_by_cluster[i])]
            y_data = [y[y_column] for j, y in enumerate(point_by_cluster[i])]
            plt.scatter(x_data, y_data, label=f'Cluster {i+1}')

        plt.plot(
                centroids_x,
                centroids_y,
                c='white',
                marker='.',
                linewidth='0.01',
                markerfacecolor='black',
                markersize='22')
        plt.plot((4.2, 6.5), (3.5, 1))
        plt.plot((4.2, 8), (7, 3.8))
        plt.title("K-Means Clustering")
        plt.xlabel(df.columns[x_column])
        plt.ylabel(df.columns[y_column])
        plt.legend()

        plt.savefig(f'figs/decision_boundary-{self.k}.png')
