from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def split(data, percent):
    train, test = train_test_split(data, test_size=percent, random_state=1)
    return train, test

def classifyMP(train, test):
    cls = MLPClassifier(solver='lbfgs', alpha=0.00095, learning_rate='adaptive', learning_rate_init=0.005, max_iter=300,
                       random_state=0)
    Perceptron = cls.fit(train)
    predictions = Perceptron.predict(test)
    for i in range(len(predictions)):
        print(predictions[i])