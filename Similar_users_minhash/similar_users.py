import pandas as pd
import numpy as np

RATINGS = "ml-latest-small/ratings.csv"
MOVIES = "ml-latest-small/movies.csv"
df1 = pd.read_csv(RATINGS, usecols=['userId', 'movieId'])
df2 = pd.read_csv(MOVIES, usecols=['movieId'])

user_movie = {}
movies = {index: movie_id for index, movie_id in enumerate(df2['movieId'])} # dictionary mapping matrix indices to movieId
users = range(1, 611)
rows = len(movies)
cols = len(users)
for i, row in df1.iterrows():
    user_id = row['userId']
    movie_id = row['movieId']

    if user_id not in user_movie:
        user_movie[user_id] = set()

    user_movie[user_id].add(movie_id)

def get_user_movie_matrix(user_movie, users, movies):
    mat = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            if movies[r] in user_movie[users[c]]:
                mat[r][c] = 1
    return mat

user_movie_matrix = get_user_movie_matrix(user_movie, users, movies)
#print(user_movie_matrix[:10, :10])

def jaccard_similarity(setA, setB):
    intersection = len(setA.intersection(setB))
    union = len(setA.union(setB))
    return intersection / union if union != 0 else 0

def jaccard_similar_users():
    similar_users = []
    for user1 in user_movie:
        for user2 in user_movie:
            if user1 < user2:
                similarity = jaccard_similarity(user_movie[user1], user_movie[user2])
                if similarity >= 0.5:
                    similar_users.append((user1, user2))
    return similar_users

jaccard_similar_users = jaccard_similar_users()
#print("Similar users using Jaccard:", jaccard_similar_users)

def minhash(matrix, quantity):
    signature_matrix = np.full((quantity, cols), 9725, dtype=int)

    for r in range(rows):
        # ((2(i+1)+1) * r + 100 * (i+1) % M and M = 9742
        M = 9742
        for c in range(cols): # col: user signature, i: hash functions
            if matrix[r][c] == 1:
                for i in range(quantity):
                    hashcode = (2 * (i+1) + 1) * r + 100 * (i+1) % M
                    if signature_matrix[i][c] > hashcode:
                        signature_matrix[i][c] = hashcode

    return signature_matrix


def signature_similarity(signature_matrix):
    similarities = []
    union = len(signature_matrix)
    for user1 in range(cols - 1):
        for user2 in range(user1 + 1, cols):
            intersection = sum(sig1 == sig2 for sig1, sig2 in zip(signature_matrix[:, user1], signature_matrix[:, user2]))
            similarity = intersection / union
            if similarity >= 0.5:
                similarities.append((user1 + 1, user2 + 1))

    return similarities

signature_50 = minhash(user_movie_matrix, 50)
signature_100 = minhash(user_movie_matrix, 100)
signature_200 = minhash(user_movie_matrix, 200)

minhash50_similar_users = signature_similarity(signature_50)
minhash100_similar_users = signature_similarity(signature_100)
minhash200_similar_users = signature_similarity(signature_200)
#%%
print("Signature similarity (50):", len(minhash50_similar_users))
#%%
#print("Signature similarity (100):", signature_similarity(signature_100))
#%%
#print("Signature similarity (200):", signature_similarity(signature_200))
#%%
def results(jaccard_similar_users, minhash_similar_users):
    jaccard_similar_users = set(jaccard_similar_users)
    minhash_similar_users = set(minhash_similar_users)
    TP = len(jaccard_similar_users.intersection(minhash_similar_users))
    TN = len(set([(i, j) for i in range(1, 9) for j in range(1, 9)]) - (jaccard_similar_users.union(minhash_similar_users)))
    FP = len(minhash_similar_users - jaccard_similar_users)
    FN = len(jaccard_similar_users - minhash_similar_users)
    print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)
#%%
#print("Minhash 50 results:")
#results(jaccard_similar_users, minhash50_similar_users)
#%%
#print("Minhash 100 results:")
#results(jaccard_similar_users, minhash100_similar_users)
#%%
#print("Minhash 200 results:")
#results(jaccard_similar_users, minhash200_similar_users)