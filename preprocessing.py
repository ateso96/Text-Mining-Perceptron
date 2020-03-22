from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
import re

from getRAW import loadData


# Dejar el texto limpio
def dataCleaner(text):
    '''
    :param text: texto del csv
    :return: texto sin caracteres raros, espacios, numeros...
    '''
    wnl = WordNetLemmatizer()
    textAux = []
    for t in range(0, len(text)):
        # Eliminar caracteres especiales
        element = re.sub(r'\W', ' ', str(text[t]))

        # Eliminar caracteres unicos
        element = re.sub(r'\s+[a-zA-Z]\s+', ' ', element)

        # Eliminar caracteres unicos desde el inicio
        element = re.sub(r'\^[a-zA-Z]\s+', ' ', element)

        # Eliminar espacios multiples
        element = re.sub(r'\s+', ' ', element, flags=re.I)

        # Eliminar prefijo 'b'
        element = re.sub(r'^b\s+', '', element)

        # Eliminar los numeros
        element = re.sub(r"\d", "", element)

        # Convertir a minusculas
        element = element.lower()

        # Lemmatization
        element = element.split()

        element = [wnl.lemmatize(word) for word in element]
        element = ' '.join(element)

        textAux.append(element)

    return textAux


# Pasar de string a representacion bow en vector
def stringToBoW(text):
    '''
    :param text: Vector de Strings
    :return: Vector con la representación en 0 y 1 de las palabras que aparecen
    '''
    vectorizer = CountVectorizer(max_features=1500, min_df=10, max_df=0.75, stop_words=stopwords.words('english'))
    '''
    max_features --> tamaño del diccionario
    min_df --> minimo de apariciones de una palabra en todas las frases para tenerla en cuenta
    max_df --> una palabra debe aparecer en ese porcentaje para tenerla en cuenta
    '''
    return vectorizer.fit_transform(text).toarray()


# Pasar de bow a tdidf
def bowToTFIDF(vector):
    '''
    :param vector: vector de 1 y 0 en Bow
    :return: representacion TF
    '''
    tfidfconverter = TfidfTransformer()
    return tfidfconverter.fit_transform(vector).toarray()


def getDataVector(filePath):
    '''
    :param filePath: path del fichero que contiene los datos
    :return: el dataset en representacion tf
    '''
    id, text, labels = loadData(filePath)
    text = dataCleaner(text)
    labels = dataCleaner(labels)

    text = stringToBoW(text)
    labels = stringToBoW(labels)

    return bowToTFIDF(text), bowToTFIDF(labels)

def getDataVectorPredict(data):
    '''
    :param filePath: path del fichero que contiene los datos
    :return: el dataset en representacion tf
    '''
    text = dataCleaner(data)
    for i in range(len(text)):
        print(text[i])

    text = stringToBoW(text)

    return bowToTFIDF(text)

