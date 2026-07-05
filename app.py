from flask import Flask, request, jsonify, render_template
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Inicializar la app web
app = Flask(__name__)

# 2. Preparar el modelo (Se ejecuta una sola vez al prender el servidor)
print("Entrenando el modelo, por favor espera...")
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

# Cargar y entrenar (igual que antes)
df = pd.read_csv('financial_data.csv', encoding='latin-1', header=None, names=['texto', 'sentimiento'])
df = df[df['sentimiento'].isin(['positive', 'negative', 'neutral'])].dropna()
df['texto_limpio'] = df['texto'].apply(limpiar_texto)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['texto_limpio'])
y = df['sentimiento']

modelo = LogisticRegression(max_iter=1000, class_weight='balanced')
modelo.fit(X, y)
print("¡Modelo listo y servidor corriendo!")

# 3. Rutas de la API

# Esta ruta sirve tu interfaz visual (el HTML)
@app.route('/')
def home():
    return render_template('index.html')

# Esta ruta recibe el texto del HTML, hace la predicción y devuelve el resultado
@app.route('/predecir', methods=['POST'])
def predecir():
    datos = request.get_json()
    texto_usuario = datos['texto']
    
    # Pasamos el texto por el mismo pipeline
    texto_limpio = limpiar_texto(texto_usuario)
    vector = vectorizer.transform([texto_limpio])
    
    # Obtenemos la predicción y las probabilidades
    prediccion = modelo.predict(vector)[0]
    probabilidades = modelo.predict_proba(vector)[0]
    
    # Las probabilidades vienen en orden alfabético: ['negative', 'neutral', 'positive']
    resultado = {
        'prediccion': prediccion.upper(),
        'prob_neg': round(probabilidades[0] * 100),
        'prob_neu': round(probabilidades[1] * 100),
        'prob_pos': round(probabilidades[2] * 100)
    }
    
    return jsonify(resultado)

if __name__ == '__main__':
    app.run(debug=True)