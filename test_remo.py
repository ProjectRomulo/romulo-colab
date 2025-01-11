import unittest
import numpy as np
from romulo_colab.remo_core import similitud_coseno, extraer_palabras_clave, analizar_texto

class TestRemo(unittest.TestCase):
    #Pruebas de similitud coseno
    def test_similitud_coseno_con_vectores_cero(self):
        self.assertEqual(similitud_coseno(np.array([0,0]), np.array([0,0])), 0.0)

    def test_similitud_coseno_con_vectores_iguales(self):
        self.assertEqual(similitud_coseno(np.array([1,1]),np.array([1,1])), 1.0)

    def test_similitud_coseno_con_vectores_opuestos(self):
        self.assertEqual(similitud_coseno(np.array([1,1]),np.array([-1,-1])), -1.0)

    def test_similitud_coseno_con_vectores_diferentes(self):
        vector1 = np.array([1, 2, 3])
        vector2 = np.array([4, 5, 6])
        resultado_esperado = 0.992277872233
        self.assertAlmostEqual(similitud_coseno(vector1, vector2), resultado_esperado, places=7)

    def test_extraer_palabras_clave_texto_vacio(self):
      self.assertEqual(extraer_palabras_clave(""),[])

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
