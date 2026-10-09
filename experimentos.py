"""Experimentos reproducibles del proyecto Quadtree."""
import argparse
from pathlib import Path
from time import perf_counter
from statistics import median
import numpy as np
import pandas as pd
from quadtree import (construir_quadtree, reconstruir_quadtree, guardar_quadtree, cargar_quadtree, obtener_metricas_arbol)

def generar_imagen_prueba(tipo, tamaño, semilla=42):
    """
    Genera imágenes RGB para experimentación.

    Tipos:
    - uniforme
    - regiones
    - aleatoria
    """

    if tamaño <= 0:
        raise ValueError("El tamaño debe ser positivo")

    if tipo == "uniforme":
        imagen = np.zeros(
            (tamaño, tamaño, 3),
            dtype=np.uint8
        )
        imagen[:] = [100, 150, 200]

    elif tipo == "regiones":
        imagen = np.zeros(
            (tamaño, tamaño, 3),
            dtype=np.uint8
        )

        mitad = tamaño // 2

        imagen[:mitad, :mitad] = [255, 0, 0]
        imagen[:mitad, mitad:] = [0, 255, 0]
        imagen[mitad:, :mitad] = [0, 0, 255]
        imagen[mitad:, mitad:] = [255, 255, 0]

    elif tipo == "aleatoria":
        generador = np.random.default_rng(semilla)

        imagen = generador.integers(
            0, 256,
            size=(tamaño, tamaño, 3),
            dtype=np.uint8
        )

    else:
        raise ValueError(
            "Tipo no válido. Use uniforme, regiones o aleatoria."
        )

    return imagen

from time import perf_counter
from statistics import median

def medir_tiempos(imagen, tolerancia=0, repeticiones=5):
    """
    Mide tiempos de construcción y reconstrucción.

    Devuelve las medianas en milisegundos.
    """

    if repeticiones < 1:
        raise ValueError("Debe realizarse al menos una repetición")

    alto, ancho = imagen.shape[:2]

    tiempos_construccion = []
    tiempos_reconstruccion = []

    for _ in range(repeticiones):

        inicio = perf_counter()

        arbol = construir_quadtree(
            imagen, 0, 0,
            ancho, alto,
            tolerancia
        )

        fin = perf_counter()

        tiempos_construccion.append(
            (fin - inicio) * 1000
        )

        inicio = perf_counter()

        reconstruir_quadtree(arbol)

        fin = perf_counter()

        tiempos_reconstruccion.append(
            (fin - inicio) * 1000
        )

    return {
        "construccion_ms": median(tiempos_construccion),
        "reconstruccion_ms": median(tiempos_reconstruccion)
    }

def ejecutar_experimentos(destino="resultados/resultados_reproducidos.csv", repeticiones=5):
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    resultados = []
    for tipo in ("uniforme", "regiones", "aleatoria"):
        for tamano in (64, 128, 256):
            imagen = generar_imagen_prueba(tipo, tamano, semilla=42)
            alto, ancho = imagen.shape[:2]
            for tolerancia in (0, 2000, 5000):
                tiempos = medir_tiempos(imagen, tolerancia=tolerancia, repeticiones=repeticiones)
                arbol = construir_quadtree(imagen, 0, 0, ancho, alto, tolerancia)
                # Archivo temporal para medir el tamaño real del formato QTR.
                ruta_qtr = destino.parent / f"prueba_{tipo}_{tamano}_tol_{tolerancia}.qtr"
                guardar_quadtree(arbol, ruta_qtr)
                recuperada = reconstruir_quadtree(cargar_quadtree(ruta_qtr))
                assert recuperada.shape == imagen.shape
                if tolerancia == 0:
                    assert np.array_equal(imagen, recuperada)
                metricas = obtener_metricas_arbol(arbol)
                bytes_originales = imagen.nbytes
                bytes_qtr = ruta_qtr.stat().st_size
                mse = float(np.mean((imagen.astype(np.float64)-recuperada.astype(np.float64))**2))
                resultados.append(dict(
                    imagen=tipo, ancho=ancho, alto=alto, pixeles=ancho*alto,
                    tolerancia=tolerancia, nodos=metricas["nodos_totales"],
                    hojas=metricas["hojas"], profundidad=metricas["profundidad_maxima"],
                    construccion_ms=tiempos["construccion_ms"],
                    reconstruccion_ms=tiempos["reconstruccion_ms"],
                    bytes_originales=bytes_originales, bytes_qtr=bytes_qtr,
                    razon_compresion=bytes_originales/bytes_qtr,
                    ahorro_porcentaje=(1-bytes_qtr/bytes_originales)*100,
                    mse=mse))
                ruta_qtr.unlink()
                print(f"{tipo} {tamano}x{tamano} tolerancia={tolerancia}: OK", flush=True)
    df = pd.DataFrame(resultados)
    df.to_csv(destino, index=False, encoding="utf-8-sig")
    print(f"Generados {len(df)} registros en {destino}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproducir experimentos Quadtree")
    parser.add_argument("--salida", default="resultados/resultados_reproducidos.csv")
    parser.add_argument("--repeticiones", type=int, default=5)
    args = parser.parse_args()
    ejecutar_experimentos(args.salida, args.repeticiones)
