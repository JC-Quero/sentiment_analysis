import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Recursos de NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    palabras = texto.split()
    palabras_limpias = [p for p in palabras if p not in stop_words]
    palabras_lematizadas = [lemmatizer.lemmatize(p) for p in palabras_limpias]
    return ' '.join(palabras_lematizadas)

# 2. Cargar el dataset real (CORREGIDO)
print("Cargando el dataset desde CSV...")
try:
    # Invertimos el orden de las columnas: primero 'texto', luego 'sentimiento'
    df = pd.read_csv('financial_data.csv', encoding='latin-1', header=None, names=['texto', 'sentimiento'])
    
    # Filtramos para asegurarnos de quedarnos solo con las filas que tienen las etiquetas correctas
    df = df[df['sentimiento'].isin(['positive', 'negative', 'neutral'])]
    df = df.dropna()
    print(f"¡Dataset cargado correctamente! Total de noticias a analizar: {len(df)}")
except FileNotFoundError:
    print("Error: No se encontró el archivo 'financial_data.csv'.")
    exit()

# 3. Aplicamos la limpieza
print("Limpiando textos (esto puede tardar unos segundos)...")
df['texto_limpio'] = df['texto'].apply(limpiar_texto)

# 4. Vectorización (TF-IDF)
print("Vectorizando el vocabulario completo...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['texto_limpio'])
y = df['sentimiento'] # Ahora sí, 'y' son las etiquetas ('positive', 'negative', 'neutral')

# 5. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Entrenamiento del Modelo
print("Entrenando el modelo...")
modelo = LogisticRegression(max_iter=1000, class_weight='balanced')
modelo.fit(X_train, y_train)

# 7. Evaluación Real
print("\n--- Resultados Reales del Modelo ---")
predicciones = modelo.predict(X_test)
print(f"Precisión (Accuracy): {accuracy_score(y_test, predicciones):.2f}")
print("\nReporte de Clasificación:")
print(classification_report(y_test, predicciones, zero_division=0))

# 8. Prueba en vivo
titular_prueba = ["Inflation rates exceed expectations, driving the tech sector stocks down."]
titular_limpio = [limpiar_texto(t) for t in titular_prueba]
titular_vectorizado = vectorizer.transform(titular_limpio)

prediccion_nueva = modelo.predict(titular_vectorizado)
print(f"\n--- Prueba en Vivo ---")
print(f"Titular original: '{titular_prueba[0]}'")
print(f"Predicción del modelo: {prediccion_nueva[0]}")