import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import json


BYTES_TO_MB_DIV = 0.000001

df = pd.read_csv('dutch-news-articles.csv')
categories = df['category'].unique()


def print_memory_usage_of_data_frame(df):
    mem = round(df.memory_usage().sum() * BYTES_TO_MB_DIV, 3)
    print("Memory usage is " + str(mem) + " MB")


def mytokenizer(text):
    return text.split()


def create_sparse_matrix(df, vocab):
    vectorizer = CountVectorizer(
        vocabulary=vocab, tokenizer=mytokenizer, lowercase=False)

    X = vectorizer.fit_transform(df['content'])

    count_vect_df = pd.DataFrame(
        X.todense(), columns=vectorizer.get_feature_names())

    def convert_to_sparse_pandas(df, exclude_columns=[]):
        """
        Converts columns of a data frame into SparseArrays and returns the data frame with transformed columns.
        Use exclude_columns to specify columns to be excluded from transformation.
        :param df: pandas data frame
        :param exclude_columns: list
            Columns not be converted to sparse
        :return: pandas data frame
        """
        df = df.copy()
        exclude_columns = set(exclude_columns)

        for (columnName, columnData) in df.iteritems():
            if columnName in exclude_columns:
                continue
            df[columnName] = pd.arrays.SparseArray(
                columnData.values, dtype='uint8')

        return df

    sparse_matrix_df = convert_to_sparse_pandas(count_vect_df)
    return sparse_matrix_df


for category in categories:
    with open('./named-entities/' + category + ".json") as json_file:
        vocab = json.load(json_file)
        print(category)

        sparse_matrix = create_sparse_matrix(df, vocab)
        df = pd.concat([df, sparse_matrix], axis=1)


del df['content']
del df['datetime']
del df['title']
print(df.head())

df.to_csv('nos-sparse-matrix.csv')

print_memory_usage_of_data_frame(df)
