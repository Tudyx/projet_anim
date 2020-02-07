import numpy as np

from collections import Counter

class DiscriminantAnalysis:
    def __init__(type="LDA"):
        pass

    def fit(X, y):
        self.n_classes = len(Counter(y.values()))
        n_samples, n_features = X.shape

        params_per_class = {
            "Class{}".format(i) for i in range(self.n_classes)
        }
        for clss in range(self.n_classes):
            indices_class = np.where(y == i)[0]
            prior = len(indices_class) / len(y)
            mean = np.mean(X[indices], axis=0)
            cov = np.cov

            params_per_class["Class{}".format(i)] = {
                "Prior": prior,
                "Mean": np.zeros([n_features, 1]),
                "Cov": np.zeros([n_features, n_features])
            }



    def predict(X):
        pass