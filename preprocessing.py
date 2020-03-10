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
    vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    return vectorizer.fit_transform(text).toarray()

# Pasar de bow a tdidf
def bowToTFIDF(vector):
    tfidfconverter = TfidfTransformer()
    return tfidfconverter.fit_transform(vector).toarray()

id, text, labels = loadData("data/train.csv")
text = dataCleaner(text)
for pos in range(len(id)):
    print(text[pos])

bowVector = stringToBoW(text)
for pos in range(len(id)):
    print(bowVector[pos])

tfidfVector = bowToTFIDF(bowVector)
for pos in range(len(id)):
    print(tfidfVector[pos])
