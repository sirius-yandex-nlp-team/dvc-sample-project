import os
import sys
import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sparse
import yaml
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

params = yaml.safe_load(open('params.yaml'))['featurize']

np.set_printoptions(suppress=True)

if len(sys.argv) != 3 and len(sys.argv) != 5:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython featurization.py data-dir-path features-dir-path\n'
    )
    sys.exit(1)

train_input = os.path.join(sys.argv[1], 'train.csv')
dev_input = os.path.join(sys.argv[1], 'dev.csv')
test_input = os.path.join(sys.argv[1], 'test.csv')

train_output = os.path.join(sys.argv[2], 'train.pkl')
dev_output = os.path.join(sys.argv[2], 'dev.pkl')
test_output = os.path.join(sys.argv[2], 'test.pkl')

max_features = params['max_features']
ngrams = params['ngrams']


def get_df(data):
    df = pd.read_csv(
        data,
        encoding='utf-8',
        header=None,
        delimiter='\t',
        names=['id', 'original', 'edit', 'scores', 'meanGrade']
    )
    sys.stderr.write(f'The input data frame {data} size is {df.shape}\n')
    return df


def replace_anchors(original, edit):
    pattern = '<\w*/>'
    found = re.search(pattern, original)
    if found:
        return original.replace(found.group(0), edit)
    
    return original


def parse_df(df_):
    # hardcode it for now
    df = df_.copy(deep=True)
    df = df['id', 'original', 'edit', 'meanGrade']

    # search for replaced token with regex
    # todo try except
    pattern = '<\w*/>'
    df['edited'] = df.apply(lambda row: replace_anchors(row['original'], row['edit']), axis=1)
    df = df.drop(columns=['original', 'edit'])


def save_matrix(df, matrix, output):
    id_matrix = sparse.csr_matrix(df['id'].astype(np.int64)).T
    target_matrix = sparse.csr_matrix(df['meanGrade'].astype(np.int64)).T

    result = sparse.hstack([id_matrix, target_matrix, matrix], format='csr')

    msg = 'The output matrix {} size is {} and data type is {}\n'
    sys.stderr.write(msg.format(output, result.shape, result.dtype))

    with open(output, 'wb') as fd:
        pickle.dump(result, fd, pickle.HIGHEST_PROTOCOL)
    pass


os.makedirs(sys.argv[2], exist_ok=True)

# Generate train feature matrix
df_train = get_df(train_input)
train_words = np.array(df_train.text.str.lower().values.astype('U'))

bag_of_words = CountVectorizer(
    stop_words='english',
    max_features=max_features,
    ngram_range=(1, ngrams)
)

bag_of_words.fit(train_words)
train_words_binary_matrix = bag_of_words.transform(train_words)
tfidf = TfidfTransformer(smooth_idf=False)
tfidf.fit(train_words_binary_matrix)
train_words_tfidf_matrix = tfidf.transform(train_words_binary_matrix)

save_matrix(df_train, train_words_tfidf_matrix, train_output)

# Generate test feature matrix
df_test = get_df(test_input)
test_words = np.array(df_test.text.str.lower().values.astype('U'))
test_words_binary_matrix = bag_of_words.transform(test_words)
test_words_tfidf_matrix = tfidf.transform(test_words_binary_matrix)

save_matrix(df_test, test_words_tfidf_matrix, test_output)
