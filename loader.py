import csv

def loadData(file):
    with open(file, encoding="utf8") as csv_file:
        reader = csv.reader(csv_file, delimiter = ",", quotechar = '"')
        id = []
        text = []
        label = []
        for row in reader:
            id.append(row[0])
            text.append(row[1])
            label.append(row[2])
        return id[1:], text[1:], label[1:]

id, text, labels = loadData("data/train.csv")
for pos in range(len(id)):
    print(labels[pos])