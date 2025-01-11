import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

try:
    nltk.data.find('wordnet.zip')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

class KeywordExtractionError(Exception):
    """Excepción personalizada para errores en la extracción de palabras clave."""
    pass

def similitud_coseno(vec1, vec2):
    """Calcula la similitud coseno entre dos vectores.

    Args:
        vec1 (numpy.ndarray): El primer vector.
        vec2 (numpy.ndarray): El segundo vector.

    Returns:
        float: La similitud coseno entre los vectores, o None si ocurre un error.
    """
    try:
        if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
            raise TypeError("Los argumentos deben ser arrays de NumPy.")

        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0

        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    except Exception as e:
        print(f"Error en similitud_coseno: {e}")
        return None

def extraer_palabras_clave(texto, use_embeddings=True, embedding_model="all-MiniLM-L6-v2"):
  """Extrae palabras clave de un texto utilizando embeddings o TF-IDF.

    Args:
        texto (str): El texto del que se extraerán las palabras clave.
        use_embeddings (bool, opcional): Indica si se deben usar embeddings. Por defecto es True.
        embedding_model (str, opcional): El modelo de embeddings a utilizar. Por defecto es "all-MiniLM-L6-v2".

    Returns:
        list or numpy.ndarray: Una lista de tuplas (palabra, score) si se usa TF-IDF, o un array de NumPy con los embeddings si se usan embeddings.

    Raises:
        KeywordExtractionError: Si ocurre un error durante la extracción.
    """
  try:
        stop_words = set(stopwords.words('spanish'))
        lemmatizer = WordNetLemmatizer()
        texto_procesado = ' '.join([lemmatizer.lemmatize(word.lower()) for word in texto.split() if word.lower() not in stop_words])

        if use_embeddings:
            try:
                embedder = SentenceTransformer(embedding_model)
                embeddings = embedder.encode([texto_procesado])
                return embeddings
            except Exception as e:
                print(f"Error al cargar/usar modelo de embeddings {embedding_model}: {e}. Usando TF-IDF")
                use_embeddings = False

        if not use_embeddings:
            vectorizer = TfidfVectorizer()
            vectorizer.fit([texto_procesado])
            tfidf_scores = vectorizer.transform([texto_procesado]).toarray()[0]
            palabras = vectorizer.get_feature_names_out()
            palabras_clave = []
            for i, palabra in enumerate(palabras):
                score = tfidf_scores[i]
                palabras_clave.append((palabra, score))
            return palabras_clave

  except Exception as e:
        raise KeywordExtractionError(f"Error al extraer palabras clave: {e}")

def analizar_texto(texto):
    print(f"Analizando el texto: {texto}")
    # Aquí se implementará el análisis de texto en futuras versiones
    return None

def registrar_interaccion(texto, emocion, hablante, memoria):
    """Registra una interacción en la memoria."""
    try:
        memoria.registrar_interaccion(texto, emocion, hablante)
    except AttributeError as e:
        print(f"Error al registrar interaccion, la memoria no esta inicializada correctamente: {e}")
    except Exception as e:
        print(f"Error al registrar la interaccion: {e}")
