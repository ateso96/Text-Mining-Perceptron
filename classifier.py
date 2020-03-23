import pickle
import numpy as np
from random import uniform

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, \
    multilabel_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.utils.optimize import _check_optimize_result


def split(data, labels, percent):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=percent, random_state=0)
    return x_train, x_test, y_train, y_test


def classifyMP(x_train, x_test, y_train, y_test):
    cls = MLPClassifier()
    parameter_space = {
        'max_iter': [1000, 5000, 10000],
        'hidden_layer_sizes': [(100, 10), (200, 20), (300, 30), (400, 40), (500, 50), (600, 60), (700, 70), (800, 80), (900, 90), (1000, 100)],
        'solver': ['lbfgs'],
        'alpha': 10.0 ** -np.arange(1, 5),
        'learning_rate': ['adaptive'],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'random_state': [0]
    }

    perceptron = GridSearchCV(cls, parameter_space, n_jobs=-1)
    perceptron.fit(x_train, y_train)

    # Best paramete set
    print("Score: ", perceptron.score(x_train, y_train))
    print('Best parameters found:\n', perceptron.best_params_)

    predictions = perceptron.predict(x_test)
    a = accuracy_score(y_test, predictions)
    p = precision_score(y_test, predictions, average='weighted')
    r = recall_score(y_test, predictions, average='weighted')
    print(a)
    print(p)
    print(r)
    print(multilabel_confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    with open('text_classifier', 'wb') as picklefile:
        pickle.dump(perceptron, picklefile)

def makePredictions(data):
    with open('text_classifier', 'rb') as training_model:
        model = pickle.load(training_model)
    model.predict(data)