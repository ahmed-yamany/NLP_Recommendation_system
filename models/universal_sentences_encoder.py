import tensorflow as tf
import tensorflow_hub as t_hub
from models.model import Model
from sklearn.feature_extraction.text import TfidfVectorizer


class UniversalSentenceEncoder(Model):
    def __init__(self, path=None):
        """
        :param path: folder path of trained model downloaded from tensorflow hub
        The Universal Sentence Encoder encodes text into high-dimensional vectors
        that can be used for text classification, semantic similarity,
        clustering and other natural language tasks.
        https://tfhub.dev/google/universal-sentence-encoder/4
        """

        self.path = path
        self.model = self.load_model()

    def load_model(self):
        if self.path is None:
            return t_hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        return tf.keras.models.load_model(self.path)

    def embed(self, texts: [str]):
        """
        :param texts: is variable length English text
        :return: 512 dimensions per sentence
        """
        return self.model(texts)

    def features(self, texts: [str]):
        return self.embed(texts)


class TVectorizer(TfidfVectorizer, Model):
    def __init__(self):
        super().__init__()

    def features(self, texts: [str]):
        return self.fit_transform(texts)

    def embed(self, texts: [str]):
        return self.transform(texts)
