import pickle

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, \
    multilabel_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier


def split(data, labels, percent):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=percent, random_state=0)
    return x_train, x_test, y_train, y_test


def classifyMP(x_train, x_test, y_train, y_test):
    cls = MLPClassifier()
    parameter_space = {
        'max_iter': [1000],
        'hidden_layer_sizes': [(100, 10)],
        'solver': ['lbfgs'],
        'alpha': 10.0 ** -np.arange(1),
        'learning_rate': ['adaptive'],
        'activation': ['identity'],
        'random_state': [0]
    }

    perceptron = GridSearchCV(cls, parameter_space, n_jobs=-1)
    perceptron.fit(x_train, y_train)

    # Best paramete set
    print("\n--> Score: ", perceptron.score(x_train, y_train))
    print('--> Best parameters:\n', perceptron.best_params_, '\n')

    predictions = perceptron.predict(x_test)
    print("\n--> Accuraccy: ", accuracy_score(y_test, predictions))
    print("\n--> Precision: ", precision_score(y_test, predictions, average='weighted'))
    print("\n--> Recall: ", recall_score(y_test, predictions, average='weighted'))

    print("\n--> Classification Report: \n",classification_report(y_test, predictions))

    print("\n--> Confusion Matrix: \n",multilabel_confusion_matrix(y_test, predictions))

    with open('modeloPerceptron', 'wb') as picklefile:
        pickle.dump(perceptron, picklefile)

def makePredictions(data):
    with open('modeloPerceptron', 'rb') as training_model:
        model = pickle.load(training_model)
    model.predict(data)