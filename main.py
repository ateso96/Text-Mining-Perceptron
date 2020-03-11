from preprocessing import getDataVector
from classifier import split, classifyMP

file = "data/train.csv"

data = getDataVector(file)

x_train, x_test, y_train, y_test = split(data, 0.7)

classifyMP(x_train, x_test, y_train, y_test)
