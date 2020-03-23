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


# Pasar de bow a tdidf
def getDictionary(vector):
    '''
    :param vector: vector Bow
    :return: diccionario
    '''
    dictionary = TfidfVectorizer(max_features=5000)
    dictionary.fit(vector)
    return dictionary

