import csv


# Leer el CSV y separar los datos
def loadData(file):
    with open(file, encoding="utf8") as csvFile:
        reader = csv.reader(csvFile, delimiter=",", quotechar='"')
        id = []
        text = []
        label = []
        for row in reader:
            id.append(row[0])  # Añadir los ID
            text.append(row[1])  # Añadir las preguntas
            label.append(row[2])  # Indicar la clasificación de cada pregunta
        return id[1:], text[1:], label[1:]


id, text, labels = loadData("data/train.csv")
for pos in range(len(id)):
    print(labels[pos])
