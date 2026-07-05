import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    palabras = [lemmatizer.lemmatize(p) for p in texto.split() if p not in stop_words]
    return ' '.join(palabras)

print("Entrenando por última vez...")
df = pd.read_csv('financial_data.csv', encoding='latin-1', header=None, names=['texto', 'sentimiento'])
df = df[df['sentimiento'].isin(['positive', 'negative', 'neutral'])].dropna()
df['texto_limpio'] = df['texto'].apply(limpiar_texto)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['texto_limpio'])
y = df['sentimiento']

modelo = LogisticRegression(max_iter=1000, class_weight='balanced')
modelo.fit(X, y)

# ¡LA MAGIA OCURRE AQUÍ! Guardamos el cerebro del modelo en archivos físicos
joblib.dump(modelo, 'modelo.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("¡Archivos modelo.pkl y vectorizer.pkl creados con éxito!")