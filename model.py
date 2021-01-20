import numpy as np
from scipy.sparse import lil_matrix
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

BYTES_TO_MB_DIV = 0.000001

df = pd.read_csv('nos-sparse-matrix-small.csv')
print(df.info())

df = df.loc[df['category'] != 'Tech']
df = df.loc[df['category'] != 'Opmerkelijk']
df = df.loc[df['category'] != 'Cultuur & Media']

y = df['category']
X = df[df.columns.difference(
    ['category', 'url', 'Unnamed: 0'])]


def data_frame_to_scipy_sparse_matrix(df):
    """
    Converts a sparse pandas data frame to sparse scipy csr_matrix.
    :param df: pandas data frame
    :return: csr_matrix
    """
    arr = lil_matrix(df.shape, dtype=np.float32)
    for i, col in enumerate(df.columns):
        ix = df[col] != 0
        arr[np.where(ix), i] = 1

    return arr.tocsr()


def get_csr_memory_usage(matrix):
    mem = (X_csr.data.nbytes + X_csr.indptr.nbytes +
           X_csr.indices.nbytes) * BYTES_TO_MB_DIV
    print("Memory usage is " + str(mem) + " MB")


y_csr = y
X_csr = data_frame_to_scipy_sparse_matrix(X)
get_csr_memory_usage(X_csr)


vector_dict = {
    # 'Pandas dataframe': [X, y],
    'Scipy sparse matrix': [X_csr, y_csr]}

for key, item in vector_dict.items():

    start = time.time()
    X_train, X_test, y_train, y_test = train_test_split(
        item[0], item[1], test_size=0.3, random_state=42)

    end = time.time()
    duration = round(end-start, 2)
    print("Key:", key)
    print("Train-test split: " + str(duration) + " secs")

    start = time.time()
    model = LogisticRegression(
        random_state=0, multi_class='ovr', solver='liblinear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    end = time.time()
    duration = round(end-start, 2)
    print("Training: " + str(duration) + " secs")
    print("\n")
