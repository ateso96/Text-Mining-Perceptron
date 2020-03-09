from csv import reader

# Metodo para leer los csv con datos
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Crear las features

# Crear los pesos para las instancias de test

filename = 'train.csv'
dataset = load_csv(filename)
headers = dataset[0]
print(headers)

for i in range(1, len(dataset)):
    print(i)