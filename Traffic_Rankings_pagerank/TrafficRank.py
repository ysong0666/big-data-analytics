import pandas as pd
from scipy.sparse import coo_matrix
import numpy as np


df = pd.read_csv("chicago-taxi-rides.csv").dropna()

n = 77
shape = (n, n)

df['pickup_community_area'] = df['pickup_community_area'] - 1
df['dropoff_community_area'] = df['dropoff_community_area'] - 1
# Create a COO matrix using SciPy
matrix = coo_matrix((df['trips'].values, (df['dropoff_community_area'].values, df['pickup_community_area'].values)), shape=shape)
matrix = matrix.todense()
column_sums = np.sum(matrix, axis=0)
matrix = matrix / column_sums
print("Transition matrix:")
print(matrix)

initial_rank = np.ones((n, 1)) / n

decay_factor = 0.85
iterations = [2, 5, 10, 25, 50]

for iteration in iterations:
    rank = initial_rank
    # print(rank.shape)
    for _ in range(iteration):
        rank = (1 - decay_factor) / n + decay_factor * np.dot(matrix, rank)
    print(f'Rankings after {iteration} iterations:')
    print(rank.T)
    if (iteration == 50):
        print("High inverse of hardship index", rank[5], rank[6], rank[7], rank[31], rank[32])
        print("Low inverse of hardship index", rank[53], rank[67], rank[29], rank[25], rank[57])
