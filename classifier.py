import pickle

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, \
    multilabel_confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


def split(data, labels, percent):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=percent, random_state=42)
    return x_train, x_test, y_train, y_test


def classifyMP(x_train, x_test, y_train, y_test):
    cls = MLPClassifier()
    parameter_space = {
        'max_iter': [2500, 3000],
        'hidden_layer_sizes': [(32, 16), (16, 9), (15, 8), 7, 8, 9, 10],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'alpha': 10 ** -np.arange(1, 5)
    }

    clf = GridSearchCV(cls, parameter_space, n_jobs=-1, cv=10, scoring='f1_weighted')
    clf.fit(x_train, y_train)

    parameters = clf.best_params_

    results = "************************************" \
              "\nPERCEPTRON TREC-6 CLASSIFIER" \
              "\n************************************"
    results += "\n--> Best F1 Score: " + str(clf.best_score_)
    results += '\n--> Best parameters:\n' + str(clf.best_params_) + '\n'

    perceptron = MLPClassifier(max_iter=parameters['max_iter'], random_state=parameters['random_state'],
                               hidden_layer_sizes=parameters['hidden_layer_sizes'], solver=parameters['solver'],
                               alpha=parameters['alpha'])
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


def makePredictions(data):
    with open('results/modeloPerceptron', 'rb') as training_model:
        model = pickle.load(training_model)
    return model.predict(data)
