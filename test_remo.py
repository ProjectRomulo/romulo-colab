import unittest
import numpy as np
from romulo_colab.remo_core import similitud_coseno, extraer_palabras_clave, analizar_texto

class TestRemo(unittest.TestCase):
    # ... (resto de las pruebas)
    def test_extraer_palabras_clave_con_texto(self):
      texto = "Este es un texto de prueba"
      palabras_clave = extraer_palabras_clave(texto)
      self.assertIsInstance(palabras_clave, list)

    def test_extraer_palabras_clave_con_embeddings(self):
      texto = "Este texto se va a convertir en embeddings"
      embeddings = extraer_palabras_clave(texto, use_embeddings=True)
      self.assertIsInstance(embeddings, np.ndarray)

    def test_extraer_palabras_clave_sin_embeddings(self):
      texto = "Este texto se analizará con TF-IDF"
      palabras_clave = extraer_palabras_clave(texto, use_embeddings=False)
      self.assertIsInstance(palabras_clave, list)

    def test_analizar_texto(self):
        texto = "Texto de prueba para analizar."
        self.assertIsNone(analizar_texto(texto)) #Debe retornar None porque es un placeholder

if __name__ == '__main__':
    unittest.main()
