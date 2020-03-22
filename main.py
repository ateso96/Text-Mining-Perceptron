import pickle

from preprocessing import getDataVector, loadData, getDataVectorPredict
from classifier import split, classifyMP, makePredictions

file = "data/train_small.csv"
id, dataNorm, lab = loadData(file)
data, labels = getDataVector(file)
x_train, x_test, y_train, y_test = split(data, lab, 0.7)
classifyMP(x_train, x_test, y_train, y_test)

filePreds = "data/test_unk.csv"
id, dataPreds, labelsPreds = loadData(filePreds)
preds = getDataVectorPredict(dataPreds)
res = makePredictions(preds)
for i in range(len(id)):
    print("X=%s, Predicted=%s" % (id[i], res[i]))