import pickle

from preprocessing import getDataVector
from classifier import split, classifyMP

file = "data/train.csv"
id, text, labels =
#data, labels = getDataVector(file)

#x_train, x_test, y_train, y_test = split(data, labels, 0.7)

#classifyMP(x_train, x_test, y_train, y_test)

with open('text_classifier', 'rb') as training_model:
    model = pickle.load(training_model)

file2 = "data/test_unk.csv"
id2, text2, labels2 =
