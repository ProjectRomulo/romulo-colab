class Memoria:
    def __init__(self):
        self.memoria = {}

    def registrar_interaccion(self, texto, emocion, hablante):
        if hablante not in self.memoria:
            self.memoria[hablante] = []
        self.memoria[hablante].append({"texto": texto, "emocion": emocion})

    def obtener_memoria(self, hablante):
        return self.memoria.get(hablante, [])
