import pickle

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, \
    multilabel_confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


def split(data, labels, percent):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=percent, random_state=42)
    return x_train, x_test, y_train, y_test


def classifyMP(x_train, x_test, y_train, y_test):
    cls = MLPClassifier()
    parameter_space = {
        'hidden_layer_sizes': [(10, 12, 6), (8, 10, 6)],
        'random_state': [1],
        'learning_rate_init': 10.0 ** -np.arange(1, 5),
        'verbose': [True],
        'activation': ['logistic', 'relu'],
        'alpha': 10.0 ** -np.arange(1, 5),
        'tol': [1e-2, 1e-3],
        'early_stopping': [True],
        'solver': ['sgd', 'adam'],
        'learning_rate': ['constant','invscaling','adaptive']
    }

    clf = GridSearchCV(cls, parameter_space, n_jobs=-1, cv=10, scoring='accuracy')
    clf.fit(x_train, y_train)

    parameters = clf.best_params_

    results = "************************************" \
              "\nPERCEPTRON TREC-6 CLASSIFIER" \
              "\n************************************"
    results += "\n--> Best F1 Score: " + str(clf.best_score_)
    results += '\n--> Best parameters:\n' + str(clf.best_params_) + '\n'

    perceptron = MLPClassifier(random_state=1, learning_rate_init=parameters['learning_rate_init'],
                               hidden_layer_sizes=parameters['hidden_layer_sizes'], shuffle=True, verbose=True,
                               activation=parameters['activation'], alpha=parameters['alpha'],
                               tol=parameters['tol'], early_stopping=True, solver=parameters['solver'],
                               learning_rate=parameters['learning_rate_init'])

    perceptron.fit(x_train, y_train)
    predictions = perceptron.predict(x_test)

    results += "\n--> Accuraccy: " + str(accuracy_score(y_test, predictions))
    results += "\n--> Precision: " + str(precision_score(y_test, predictions, average='weighted'))
    results += "\n--> Recall: " + str(precision_score(y_test, predictions, average='weighted'))

    results += "\n--> Classification Report: \n" + classification_report(y_test, predictions)

    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(perceptron, x_test, y_test,
                                     display_labels=['ABBR', 'DESC', 'ENTY', 'HUM', 'LOC', 'HUM'],
                                     cmap=plt.cm.Blues)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.savefig('results/matrix.png')

    with open('results/resultados.txt', 'wb') as picklefile:
        pickle.dump(results, picklefile)

    with open('results/modeloPerceptron', 'wb') as picklefile:
        pickle.dump(perceptron, picklefile)


def makePredictionsPerceptron(data):
    with open('results/modeloPerceptron', 'rb') as training_model:
        model = pickle.load(training_model)
    return model.predict(data)

def classifyBaseline(x_train, x_test, y_train, y_test):
    cls = DummyClassifier(strategy='stratified', random_state=1)

    results = "************************************" \
              "\nBASELINE TREC-6 CLASSIFIER" \
              "\n************************************"

    cls.fit(x_train, y_train)
    predictions = cls.predict(x_test)

    results += "\n--> Accuraccy: " + str(accuracy_score(y_test, predictions))
    results += "\n--> Precision: " + str(precision_score(y_test, predictions, average='weighted'))
    results += "\n--> Recall: " + str(precision_score(y_test, predictions, average='weighted'))

    results += "\n--> Classification Report: \n" + classification_report(y_test, predictions)

    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(cls, x_test, y_test,
                                     display_labels=['ABBR', 'DESC', 'ENTY', 'HUM', 'LOC', 'HUM'],
                                     cmap=plt.cm.Blues)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.savefig('results/matrixBaseline.png')

    with open('results/resultadosBaseline.txt', 'wb') as picklefile:
        pickle.dump(results, picklefile)

    with open('results/modeloBaseline', 'wb') as picklefile:
        pickle.dump(cls, picklefile)
