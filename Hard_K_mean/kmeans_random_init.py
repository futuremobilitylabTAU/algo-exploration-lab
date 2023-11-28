"""
K-means Clustering with Random Initialization

This script implements the K-means clustering algorithm with random initialization of cluster centers. It generates N random vectors with D dimensions, assigns them to K clusters randomly, calculates the centroids of the clusters, and visualizes the resulting clusters in 2D/3D.

Author: Gabriel Dadashev
Date: 25-11-2023
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Set the convergence threshold for K-means
epsilon=0.00001

# Initialize convergence metrics
convergence_threshold = np.inf
total_cost=np.inf


def assign_cluster_and_distance(x, D, mean_vector):
    """
    Assigns a data point to the nearest cluster center and calculates the distance.

    Parameters:
        x (pd.Series): Data point coordinates.
        D (int): Number of dimensions.
        mean_vector (pd.DataFrame): DataFrame containing cluster centers.

    Returns:
        tuple: Cluster index and distance to the assigned cluster center.
    """
    temp = mean_vector.copy()
    temp['distance'] = temp.apply(lambda y: calculate_distance(y, x), axis=1)
    temp = temp[temp['distance'] == temp['distance'].min()]
    return int(temp.index.values), float(temp['distance'])

def calculate_distance(y, x):
    """
   Calculates the Euclidean distance between two points.

   Parameters:
       y (pd.Series): Coordinates of the first point.
       x (pd.Series): Coordinates of the second point.

   Returns:
       float: Euclidean distance between the points.
   """
    summation = sum((x[i] - y[i]) ** 2 for i in range(D))
    return math.sqrt(summation)



### N random vectors with D dimensions
N=1000
D=3

data=np.random.rand(N,D)
data_df=pd.DataFrame(data=data)

# Randomly selecting K vectors as cluster centers.
K=3
mean_vactor=data_df.sample(K)
mean_vactor=mean_vactor.reset_index()
mean_vactor.index=mean_vactor.index+1


# Lists to store convergence data
iteration = 0
convergence_iterations = []
convergence_costs = []



# K-means clustering loop
while convergence_threshold>epsilon:

    # Assign clusters and calculate distances
    data_df['cluster/distance']=data_df.apply(lambda x: assign_cluster_and_distance(x,D,mean_vactor),axis=1)
    data_df['cluster']=data_df.apply(lambda x: x['cluster/distance'][0],axis=1)
    data_df['distance']=data_df.apply(lambda x: x['cluster/distance'][1],axis=1)
    data_df=data_df.drop(['cluster/distance'],axis=1)
    
    # Update  convergence threshold    
    data_df['distance']=data_df['distance']**2
    prev_total_cost=total_cost
    total_cost=data_df['distance'].sum()
    convergence_threshold=abs(total_cost-prev_total_cost)
    
    # Update cluster centers
    mean_vactor=data_df.groupby('cluster').mean()
    
    # Increment iteration counter
    iteration += 1
    
    # Append convergence data for plotting
    convergence_iterations.append(iteration)
    convergence_costs.append(total_cost)
    
    print(convergence_threshold)



### If D equals 2, we can create 2D visualization.
if D==2:
    
    # Create a scatter plot for data points with different cluster colors
    x=data_df[0]
    y=data_df[1]
    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.scatter(x, y, alpha=0.5,c=data_df['cluster'], label='Data Points')
    # Highlight cluster centroids with a different marker and color
    axes.scatter(mean_vactor[0], mean_vactor[1], marker='+',s=250,c=mean_vactor.index, label='Cluster Centroids')
    
    # Add labels, title, and legend for better interpretation
    axes.set_xlabel('X-axis')
    axes.set_ylabel('Y-axis')
    axes.set_title('K-means Clustering with Random Initialization')
    axes.legend()
    plt.show()

###3D Visualization
    
elif D == 3:
    # Create a scatter plot for data points with different cluster colors
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = data_df[0]
    y = data_df[1]
    z = data_df[2]
    ax.scatter(x, y, z, c=data_df['cluster'], marker='o', label='Data Points')
    # Highlight cluster centroids with a different marker and color
    ax.scatter(mean_vactor[0], mean_vactor[1], mean_vactor[2], marker='+', s=600, c=mean_vactor.index, label='Cluster Centroids')
    
    # Add labels, title, and legend for better interpretation
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('K-means Clustering with Random Initialization (3D)')
    ax.legend()
    plt.show()
    

    
# Convergence graph
plt.figure()
plt.plot(convergence_iterations, convergence_costs, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Total Cost')
plt.title('Convergence of K-means Clustering')
plt.show()
