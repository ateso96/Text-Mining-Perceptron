from classifier import *
from getRAW import loadData
from preprocessing import *
import csv

file = "data/train.csv"
filePreds = "data/test_unk.csv"
id, text, labels = loadData(file)
idPreds, textPreds, labelsPreds = loadData(filePreds)

data = rawToVector(text)
dictionary = getDictionary(data)

with open('results/classifier/dictionary', 'wb') as picklefile:
    pickle.dump(dictionary, picklefile)

data = tfidf(data, dictionary)

x_train, x_test, y_train, y_test = split(data, labels, 0.3)

classifyMP(x_train, x_test, y_train, y_test)

classifyBaseline(x_train, x_test, y_train, y_test)

textPredsAux = textPreds.copy()
dataPreds = rawToVector(textPreds)
dataPreds = tfidf(dataPreds, dictionary)
predictions = makePredictionsPerceptron(dataPreds)

with open('results/predictions/test_predicted.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "text", "label"])
    for i in range(len(textPredsAux)):
        writer.writerow([i, textPredsAux[i], predictions[i]])