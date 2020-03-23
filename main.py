import pickle

from getRAW import loadData
from preprocessing import *
from classifier import split, classifyMP, makePredictions

file = "data/train_small.csv"
filePreds = "data/test_unk.csv"
id, text, labels = loadData(file)
idPreds, textPreds, labelsPreds = loadData(filePreds)

data = rawToVector(text)
dictionary = getDictionary(data)

with open('dictionary', 'wb') as picklefile:
    pickle.dump(dictionary, picklefile)

data = tfidf(data, dictionary)

x_train, x_test, y_train, y_test = split(data, labels, 0.3)
# classifyMP(x_train, x_test, y_train, y_test)

makePredictions(dataPreds)
