from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
import re

from getRAW import loadData

# Dejar el texto limpio
def dataCleaner(text):
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

        ###############################################
        # DESDE AQUI TENGO DUDAS DE QUE DEBO ELIMINAR
        # Eliminar los numeros
        element = re.sub(r"\d", "", element)
        ###############################################

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
    tfidfconverter = TfidfTransformer()
    return tfidfconverter.fit_transform(vector).toarray()

id, text, labels = loadData("data/train.csv")
text = dataCleaner(text)
for pos in range(len(id)):
    print(text[pos])

bow = stringToBoW(text)
for pos in range(len(id)):
    print(bow[pos])

