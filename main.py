import pickle

from preprocessing import getDataVector, loadData, getDataVectorPredict
from classifier import split, classifyMP, makePredictions

file = "data/train_small.csv"
filePreds = "data/test_unk.csv"
id, text, labels = loadData(file)
idPreds, textPreds, labelsPreds = loadData(filePreds)

data, dataPreds = getDataVector(file, filePreds)

x_train, x_test, y_train, y_test = split(data, labels, 0.7)
classifyMP(x_train, x_test, y_train, y_test)

res = makePredictions(dataPreds)
for i in range(len(idPreds)):
    print("X=%s, Predicted=%s" % (idPreds[i], res[i]))