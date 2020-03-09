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

filename = 'train.csv'
dataset = load_csv(filename)
for i in range(len(dataset)):
    print(dataset[i])