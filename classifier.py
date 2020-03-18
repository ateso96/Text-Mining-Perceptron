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
    cls = MLPClassifier(max_iter=1500,  random_state=0, learning_rate_init=0.005)
    parameter_space = {
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [0.0001, 0.00095, 0.05],
        'learning_rate': ['adaptive'],
        'activation': ['identity', 'logistic', 'tanh', 'relu']
    }

    perceptron = GridSearchCV(cls, parameter_space, cv=3, scoring='accuracy')
    perceptron.fit(x_train, y_train)

    # Best paramete set
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
