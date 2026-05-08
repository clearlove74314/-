import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class SVMClassifier:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel=kernel, C=C, gamma=gamma, probability=True))
        ])
    
    def fit(self, X, y):
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        self.pipeline.fit(X, y)
    
    def predict(self, X):
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        return self.pipeline.predict_proba(X)
    
    def score(self, X, y):
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        return self.pipeline.score(X, y)
    
    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f)
    
    @classmethod
    def load(cls, path):
        import pickle
        with open(path, 'rb') as f:
            pipeline = pickle.load(f)
        classifier = cls()
        classifier.pipeline = pipeline
        return classifier