
from romulo_colab.remo_core import extraer_palabras_clave
import numpy as np

texto = "Este texto se analizará con TF-IDF y embeddings."
palabras_clave_tfidf = extraer_palabras_clave(texto, use_embeddings=False)
print("Palabras clave (TF-IDF):", palabras_clave_tfidf)

embeddings = extraer_palabras_clave(texto, use_embeddings=True, embedding_model="all-MiniLM-L6-v2") #Modelo especifico
print("Embeddings:", embeddings)
