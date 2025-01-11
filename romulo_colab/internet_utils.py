import requests
from bs4 import BeautifulSoup

def buscar_en_internet(consulta, num_resultados=5):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        url = f"https://www.google.com/search?q={consulta}"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        resultados = []
        for g in soup.select('.yuRUbf > a'):
            resultados.append(g['href'])
        return resultados[:num_resultados]
    except requests.exceptions.RequestException as e:
        print(f"Error al buscar en internet: {e}")
        return None
