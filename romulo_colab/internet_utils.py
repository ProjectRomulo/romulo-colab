import requests
from bs4 import BeautifulSoup

def buscar_en_internet(consulta, num_resultados=5):
    """Busca información en internet utilizando Google.

    Args:
        consulta (str): La consulta de búsqueda.
        num_resultados (int, opcional): El número máximo de resultados a devolver. Por defecto es 5.

    Returns:
        list: Una lista de URLs con los resultados de la búsqueda, o None si ocurre un error.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        url = f"https://www.google.com/search?q={consulta}"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Lanza una excepción para códigos de estado HTTP erróneos
        soup = BeautifulSoup(response.content, "html.parser")
        resultados = []
        for g in soup.select('.yuRUbf > a'):
            resultados.append(g['href'])
        return resultados[:num_resultados]
    except requests.exceptions.RequestException as e:
        print(f"Error en la solicitud HTTP: {e}")
        return None
    except Exception as e:
        print(f"Error desconocido al buscar en internet: {e}")
        return None
