import csv

# Leer el CSV y separar los datos
import pickle
import sys


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


args = sys.argv
a = loadData(args[1])
if (args[2] == "-train"):
    with open('results/preprocessing/rawDataTrain', 'wb') as picklefile:
        pickle.dump(a, picklefile)
    print("Fichero rawDataTrain creado.")
elif (args[2] == "-test"):
    with open('results/preprocessing/rawDataTest', 'wb') as picklefile:
        pickle.dump(a, picklefile)
    print("Fichero rawDataTest creado.")
else:
    print("Parametros incorrectos")
