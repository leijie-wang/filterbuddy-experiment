import random
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sentence_transformers import SentenceTransformer
from sharedsteps.utils import calculate_algorithm_metrics

class MLFilter:

    def __init__(self, model_name, rng = random.Random(3141)):
        self.model_name = model_name
        self.model = self.set_model()
        self.pipe = None
        self.embeddings = SentenceTransformer('all-MiniLM-L6-v2') # Model for embeddings

    def set_model(self):
        if self.model_name == 'Gradient Boosting':
            return GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        elif self.model_name == 'Random Forest':
            return RandomForestClassifier(n_estimators = 100, max_depth=1, random_state=0)
        elif self.model_name == 'Bayes':
            return GaussianNB()
        elif self.model_name == 'SVM':
            return svm.SVC()
        elif self.model_name == 'MLP':
            return MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        else:
            raise Exception(f'Unrecognized model name {self.model_name}')

    def train_model(self, X, y):
        """
            Train a model with the labels and print the test results on the training set
            @param force_retrain - Force the model to be replaced even if it is already trained.
        """

        # Create the training set
        X_train, y_train = X, y
        X_train = self.embeddings.encode(X_train)

        # Train the model
        if self.pipe is None:
            self.pipe = Pipeline([
                ('scaler', StandardScaler(with_mean = False)),
                (self.model_name, self.model)
            ])
            self.pipe.fit(X_train, y_train)

        y_pred = self.pipe.predict(X_train)
        performance = calculate_algorithm_metrics(y_train, y_pred)
        return {
            "prediction": y_pred.tolist(),
            "performance": performance
        }
        

    def test_model(self, X, y):
        """
            Test the model against X, y
        """

        if self.pipe is None:
            print('Classifier not trained or stale! Please retrain via .train_model()')
            return
        
        X_test, y_test = X, y
        X_test = self.embeddings.encode(X_test)
        y_pred = self.pipe.predict(X_test)
        
        performance = calculate_algorithm_metrics(y_test, y_pred)
        return {
            "prediction": y_pred.tolist(),
            "performance": performance
        }