# Compresión de imágenes mediante Quadtrees

Proyecto académico de **Divide y Vencerás** para representar imágenes RGB mediante un árbol Quadtree, reconstruirlas y almacenarlas en archivos binarios `.qtr`.

## Contenido

- `quadtree.py`: clase `NodoQuadtree`, subdivisión, construcción, reconstrucción, métricas, serialización y formato `.qtr`.
- `experimentos.py`: generador reproducible de imágenes sintéticas y medición de tiempos, tamaños y MSE.
- `proyecto_quadtree.ipynb`: desarrollo completo, análisis teórico, pruebas, gráficas y documentación.
- `resultados/`: 27 mediciones originales (CSV y Excel).
- `evidencias/`: tres comparaciones visuales PNG y sus archivos `.qtr`.
- `reporte/`: carpeta reservada para el reporte técnico PDF.

## Requisitos

Python 3.10 o superior. Desde la raíz del repositorio:

```bash
python -m pip install -r requirements.txt
```

## Ejecutar los experimentos

```bash
python experimentos.py
```

El comando genera `resultados/resultados_reproducidos.csv` con 27 filas: 3 tipos de imagen (uniforme, regiones y aleatoria), tamaños 64, 128 y 256, y tolerancias 0, 2000 y 5000. Usa semilla 42 y medianas de cinco repeticiones. Los tiempos varían según la computadora.

Para una prueba más rápida:

```bash
python experimentos.py --repeticiones 1
```

## Uso del algoritmo

```python
from pathlib import Path
from experimentos import generar_imagen_prueba
from quadtree import construir_quadtree, guardar_quadtree, cargar_quadtree, reconstruir_quadtree

imagen = generar_imagen_prueba("regiones", 64, semilla=42)
arbol = construir_quadtree(imagen, 0, 0, 64, 64, tolerancia=0)
guardar_quadtree(arbol, "ejemplo.qtr")
recuperada = reconstruir_quadtree(cargar_quadtree("ejemplo.qtr"))
```

El algoritmo trabaja con arreglos NumPy de imágenes RGB de forma `(alto, ancho, 3)`. La tolerancia se aplica al MSE de cada región. La compresión se compara con datos **RGB sin comprimir**, no con PNG ni JPEG. A tolerancia cero se obtiene reconstrucción exacta en los casos probados, pero la representación no siempre ahorra espacio.

## Notebook

Abrir `proyecto_quadtree.ipynb` en Google Colab. El notebook contiene algunas celdas de descarga específicas de Colab; los scripts `.py` pueden ejecutarse localmente sin Colab.

## Pruebas y evidencias

Las imágenes se generan automáticamente, sin necesidad de descargarlas de Internet. En `evidencias/` se incluyen los casos uniforme (tolerancia 0), regiones (tolerancia 0) y aleatoria (tolerancia 5000), cada uno con original, reconstrucción y métricas.

## Uso de IA

Se utilizó ChatGPT como apoyo para explicaciones, estructura del proyecto, generación y revisión de fragmentos de código, depuración, diseño de pruebas y documentación. Los resultados incluidos se obtuvieron mediante ejecución del programa.
