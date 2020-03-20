import pickle

from preprocessing import getDataVector
from classifier import split, classifyMP

file = "data/train.csv"
data, labels = getDataVector(file)

x_train, x_test, y_train, y_test = split(data, labels, 0.7)

classifyMP(x_train, x_test, y_train, y_test)