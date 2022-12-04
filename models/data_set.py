import pandas as pd
from nltk.stem import PorterStemmer
import nltk


class DataSet:
    def __init__(self, path):
        self.path = path
        self.frame = pd.read_csv(path, engine='python').reset_index()

    def preprocessing(self, key):
        self.lower(key)
        self.replace_not_char(key)
        self.word_tokenize(key)
        self.stemming(key)

    def lower(self, key):
        self.frame[key] = self.frame[key].map(lambda s: str(s).lower())

    def replace_not_char(self, key):
        self.frame[key] = self.frame[key].str.replace('[^\w\s]', '')

    def word_tokenize(self, key):
        self.frame[key] = self.frame[key].apply(nltk.word_tokenize)

    def stemming(self, key):
        stemmer = PorterStemmer()
        self.frame[key] = self.frame[key].apply(lambda x: [stemmer.stem(y) for y in x])

    def feature_extraction(self, key):
        self.frame[key] = self.frame[key].apply(lambda x: ' '.join(x))
