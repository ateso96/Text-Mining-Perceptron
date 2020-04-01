import pickle
import sys
from collections import defaultdict

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


# Dejar el texto limpio
def rawToVector(text):
    '''
    :param text: texto del csv
    :return: texto sin caracteres raros, espacios, numeros... en BoW
    '''
    res = text

    # Convertir a minusculas
    text = [entry.lower() for entry in text]

    # Convertir texto en array de strings
    text = [word_tokenize(entry) for entry in text]

    # Eliminar palabras raras
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for ind, entry in enumerate(text):
        textAux = []
        wl = WordNetLemmatizer()
        for word, tag in pos_tag(entry):
            if word not in stopwords.words('english') and word.isalpha():
                wordAux = wl.lemmatize(word, tag_map[tag[0]])
                textAux.append(wordAux)
        res[ind] = str(textAux)

    return res


# Obtener un diccionario
def getDictionary(vector):
    '''
    :param vector: vector Bow
    :return: diccionario
    '''
    dictionary = TfidfVectorizer(max_features=5000)
    dictionary.fit(vector)
    print("Tamaño del diccionario: ", len(dictionary.vocabulary_))
    return dictionary


# Usando el diccionario, obtener el tfidf
def tfidf(vector, dictionary):
    return dictionary.transform(vector)


# Argumento 1: Archivo input
# Argumento 2: metodo -d Saca diccionario despues de hacer el vector

args = sys.argv
if (len(args) == 2):
    with open(args[1], 'rb') as load:
        a = pickle.load(load)
    data = rawToVector(a[1])
    dictionary = getDictionary(data)

    with open('results/preprocessing/dataBoW', 'wb') as picklefile:
        pickle.dump(data, picklefile)
    print("Bag of Words guardado")
    with open('results/preprocessing/dictionary', 'wb') as picklefile2:
        pickle.dump(dictionary, picklefile2)
    print("Diccionario guardado")

    data = tfidf(data, dictionary)
    with open('results/preprocessing/dataTFIDF', 'wb') as picklefile2:
        pickle.dump(data, picklefile2)
    print("TFIDF guardado")
