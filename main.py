import pickle

from preprocessing import getDataVector, loadData
from classifier import split, classifyMP, makePredictions

file = "data/train_small.csv"
data, labels = getDataVector(file)
x_train, x_test, y_train, y_test = split(data, labels, 0.7)
classifyMP(x_train, x_test, y_train, y_test)

filePreds = "data/test_unk.csv"
id, dataPreds, labelsPreds = loadData(filePreds)
preds, l = getDataVector(filePreds)
res = makePredictions(preds)
for i in range(len(id)):
    print("X=%s, Predicted=%s" % (id[i], res[i]))