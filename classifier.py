from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def split(data, labels, percent):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=percent, random_state=1)
    return x_train, x_test, y_train, y_test

def classifyMP(x_train, x_test, y_train, y_test):
    cls = MLPClassifier(solver='lbfgs', alpha=0.00095, learning_rate='adaptive', learning_rate_init=0.005, max_iter=300,
                       random_state=0)
    Perceptron = cls.fit(x_train, y_train)
    predictions = Perceptron.predict(x_test)
    a = accuracy_score(y_test, predictions)
    p = precision_score(y_test, predictions, average='weighted')
    r = recall_score(y_test, predictions, average='weighted')