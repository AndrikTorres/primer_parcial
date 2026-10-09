"""Compresión de imágenes RGB mediante Quadtree. Extraído del notebook del proyecto."""
import struct
from pathlib import Path
import numpy as np

class NodoQuadtree:
    def __init__(self, x, y, ancho, alto, color=None):
        self.x = x
        self.y = y
        self.ancho = ancho
        self.alto = alto
        self.color = color
        self.hijos = []

    def es_hoja(self):
        return len(self.hijos) == 0

def analizar_region(imagen, x, y, ancho, alto):
    region = imagen[y:y + alto, x:x + ancho]

    if region.size == 0:
        raise ValueError("La región no puede estar vacía")

    region_float = region.astype(np.float64)

    color_promedio = np.mean(
        region_float,
        axis=(0, 1)
    )

    mse = np.mean(
        (region_float - color_promedio) ** 2
    )

    return color_promedio, float(mse)

def dividir_region(x, y, ancho, alto):
    mitad_ancho = ancho // 2
    mitad_alto = alto // 2

    regiones = [
        (x, y, mitad_ancho, mitad_alto),
        (x + mitad_ancho, y,
         ancho - mitad_ancho, mitad_alto),
        (x, y + mitad_alto,
         mitad_ancho, alto - mitad_alto),
        (x + mitad_ancho, y + mitad_alto,
         ancho - mitad_ancho, alto - mitad_alto)
    ]

    return [
        region for region in regiones
        if region[2] > 0 and region[3] > 0
    ]

def construir_quadtree(
    imagen, x, y, ancho, alto, tolerancia=0
):
    color_promedio, mse = analizar_region(
        imagen, x, y, ancho, alto
    )

    nodo = NodoQuadtree(
        x, y, ancho, alto,
        color=color_promedio
    )

    # Caso base
    if mse <= tolerancia or (ancho == 1 and alto == 1):
        return nodo

    # Dividir y resolver recursivamente
    regiones = dividir_region(x, y, ancho, alto)

    for cx, cy, ca, ch in regiones:
        hijo = construir_quadtree(
            imagen, cx, cy, ca, ch, tolerancia
        )
        nodo.hijos.append(hijo)

    return nodo

def reconstruir_quadtree(arbol):
    imagen = np.zeros(
        (arbol.alto, arbol.ancho, 3),
        dtype=np.uint8
    )

    def pintar(nodo):
        if nodo.es_hoja():
            color = np.clip(
                np.rint(nodo.color), 0, 255
            ).astype(np.uint8)

            imagen[
                nodo.y:nodo.y + nodo.alto,
                nodo.x:nodo.x + nodo.ancho
            ] = color
        else:
            for hijo in nodo.hijos:
                pintar(hijo)

    pintar(arbol)

    return imagen

def obtener_metricas_arbol(arbol):
    metricas = {
        "nodos_totales": 0,
        "nodos_internos": 0,
        "hojas": 0,
        "profundidad_maxima": 0
    }

    def recorrer(nodo, profundidad):
        metricas["nodos_totales"] += 1

        metricas["profundidad_maxima"] = max(
            metricas["profundidad_maxima"],
            profundidad
        )

        if nodo.es_hoja():
            metricas["hojas"] += 1
        else:
            metricas["nodos_internos"] += 1

            for hijo in nodo.hijos:
                recorrer(hijo, profundidad + 1)

    recorrer(arbol, 0)

    return metricas

def serializar_quadtree(nodo):
    """
    Convierte el Quadtree en bits de estructura
    y bytes de colores RGB mediante preorden.
    """

    bits = []
    colores = bytearray()

    def recorrer(actual):
        if actual.es_hoja():
            bits.append(0)

            rgb = np.clip(
                np.rint(actual.color),
                0, 255
            ).astype(np.uint8)

            colores.extend(rgb.tolist())

        else:
            bits.append(1)

            for hijo in actual.hijos:
                recorrer(hijo)

    recorrer(nodo)

    return bits, bytes(colores)


def empaquetar_bits(bits):
    """
    Agrupa ocho bits por byte.
    """

    resultado = bytearray()

    for inicio in range(0, len(bits), 8):
        grupo = bits[inicio:inicio + 8]

        valor = 0

        for bit in grupo:
            valor = (valor << 1) | bit

        valor <<= (8 - len(grupo))
        resultado.append(valor)

    return bytes(resultado)


def desempaquetar_bits(datos, cantidad_bits):
    """
    Recupera los bits de estructura.
    """

    bits = []

    for byte in datos:
        for desplazamiento in range(7, -1, -1):
            bit = (byte >> desplazamiento) & 1
            bits.append(bit)

            if len(bits) == cantidad_bits:
                return bits

    if len(bits) != cantidad_bits:
        raise ValueError(
            "Datos insuficientes para recuperar los bits"
        )

    return bits

def deserializar_quadtree(bits, colores, ancho, alto):
    """
    Recupera el Quadtree desde los bits
    de estructura y los bytes RGB.
    """

    indice_bit = 0
    indice_color = 0

    def reconstruir_nodo(x, y, w, h):
        nonlocal indice_bit, indice_color

        if w <= 0 or h <= 0:
            raise ValueError("Dimensiones inválidas")

        if indice_bit >= len(bits):
            raise ValueError("Faltan bits de estructura")

        bit = bits[indice_bit]
        indice_bit += 1

        if bit == 0:
            if indice_color + 3 > len(colores):
                raise ValueError("Faltan datos RGB")

            rgb = np.frombuffer(
                colores[indice_color:indice_color + 3],
                dtype=np.uint8
            ).astype(np.float64)

            indice_color += 3

            return NodoQuadtree(
                x, y, w, h, color=rgb
            )

        if bit != 1:
            raise ValueError("Bit de estructura inválido")

        if w == 1 and h == 1:
            raise ValueError("Un píxel no puede subdividirse")

        nodo = NodoQuadtree(
            x, y, w, h, color=None
        )

        cuadrantes = dividir_region(x, y, w, h)

        for cx, cy, ca, ch in cuadrantes:
            hijo = reconstruir_nodo(
                cx, cy, ca, ch
            )

            nodo.hijos.append(hijo)

        return nodo

    raiz = reconstruir_nodo(
        0, 0, ancho, alto
    )

    if indice_bit != len(bits):
        raise ValueError("Sobran bits de estructura")

    if indice_color != len(colores):
        raise ValueError("Sobran bytes de colores")

    return raiz

def guardar_quadtree(raiz, ruta):
    """
    Guarda un árbol en un archivo binario .qtr.
    """

    bits, colores = serializar_quadtree(raiz)
    estructura = empaquetar_bits(bits)

    cabecera = struct.pack(
        "<4sIII",
        b"QTR1",
        raiz.ancho,
        raiz.alto,
        len(bits)
    )

    with open(ruta, "wb") as archivo:
        archivo.write(cabecera)
        archivo.write(estructura)
        archivo.write(colores)

    return {
        "ruta": str(ruta),
        "bytes_cabecera": len(cabecera),
        "bytes_estructura": len(estructura),
        "bytes_colores": len(colores),
        "bytes_totales": Path(ruta).stat().st_size
    }


def cargar_quadtree(ruta):
    """
    Lee un archivo .qtr y recupera su árbol.
    """

    with open(ruta, "rb") as archivo:
        cabecera = archivo.read(16)

        if len(cabecera) != 16:
            raise ValueError("Archivo incompleto: falta cabecera")

        identificador, ancho, alto, cantidad_bits = (
            struct.unpack("<4sIII", cabecera)
        )

        if identificador != b"QTR1":
            raise ValueError("Formato de archivo no reconocido")

        if ancho == 0 or alto == 0 or cantidad_bits == 0:
            raise ValueError("Cabecera con valores inválidos")

        bytes_estructura = (cantidad_bits + 7) // 8
        estructura = archivo.read(bytes_estructura)

        if len(estructura) != bytes_estructura:
            raise ValueError("Archivo incompleto: falta estructura")

        colores = archivo.read()

    bits = desempaquetar_bits(
        estructura, cantidad_bits
    )

    return deserializar_quadtree(
        bits, colores, ancho, alto
    )
