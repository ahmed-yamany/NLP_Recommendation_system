from sklearn.neighbors import NearestNeighbors
from models.model import Model
from models.data_set import DataSet


class Recommendation_system:
    def __init__(self, model: Model, dataset: DataSet, n_neighbors=5, *args, **kwargs):
        self.model = model
        self.dataset = dataset.frame
        self.nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors, *args, **kwargs)

    def embeddings(self, key):
        titles = [str(i) for i in self.dataset[key]]
        return self.model.features(titles)

    def fit(self, key):
        embeddings = self.embeddings(key)
        self.nearest_neighbors.fit(embeddings)

    def recommend(self, text: str):
        embed = self.model.embed([text])
        # embed = self.model.transform([text])
        neighbors = self.nearest_neighbors.kneighbors(embed,n_neighbors=5, return_distance=False)[0]
        return self.dataset.iloc[neighbors]
