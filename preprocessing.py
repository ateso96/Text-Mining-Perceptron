from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from collections import defaultdict

from getRAW import loadData


# Dejar el texto limpio
def rawToVector(text):
    '''
    :param text: texto del csv
    :return: texto sin caracteres raros, espacios, numeros...
    '''
    res = text

    # Convertir a minusculas
    text = [entry.lower() for entry in text]

    # Convertir texto en array de strings
    text = [word_tokenize(entry) for entry in text]

    # Eliminar palabras raras
    tag_map = defaultdict(lambda  : wn.NOUN)
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


# Pasar de string a representacion bow en vector
def stringToBoW(text, textPreds):
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
    return vectorizer.fit_transform(text).toarray(), vectorizer.fit_transform(textPreds).toarray()


# Pasar de bow a tdidf
def bowToTFIDF(vector, vectorPreds):
    '''
    :param vector: vector de 1 y 0 en Bow
    :return: representacion TF
    '''
    tfidfconverter = TfidfTransformer()
    return tfidfconverter.fit_transform(vector).toarray(), tfidfconverter.fit_transform(vectorPreds).toarray()


def getDataVector(file, filePreds):
    '''
    :param filePath: path del fichero que contiene los datos
    :return: el dataset en representacion tf
    '''
    id, text, labels = loadData(file)
    idP, textP, labelsP = loadData(filePreds)
    text = rawToVector(text)
    textP = rawToVector(textP)

    text, textP = stringToBoW(text, textP)
    vector, vectorPreds = bowToTFIDF(text, textP)

    return vector, vectorPreds

def getDataVectorPredict(data):
    '''
    :param filePath: path del fichero que contiene los datos
    :return: el dataset en representacion tf
    '''
    text = rawToVector(data)
    for i in range(len(text)):
        print(text[i])

    text = stringToBoW(text)

    return bowToTFIDF(text)

