from flask import Flask, request, jsonify, render_template
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Descargamos los diccionarios necesarios para procesar el texto del usuario
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Cargar el modelo pre-entrenado
modelo = joblib.load('modelo.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    palabras = [lemmatizer.lemmatize(p) for p in texto.split() if p not in stop_words]
    return ' '.join(palabras)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    datos = request.get_json()
    texto_usuario = datos['texto']
    
    texto_limpio = limpiar_texto(texto_usuario)
    vector = vectorizer.transform([texto_limpio])
    
    prediccion = modelo.predict(vector)[0]
    probabilidades = modelo.predict_proba(vector)[0]
    
    resultado = {
        'prediccion': prediccion.upper(),
        'prob_neg': round(probabilidades[0] * 100),
        'prob_neu': round(probabilidades[1] * 100),
        'prob_pos': round(probabilidades[2] * 100)
    }
    
    return jsonify(resultado)

if __name__ == '__main__':
    app.run(debug=True)