# Clasificación de canciones por género en base a la música usando aprendizaje colectivo

Reconocimiento de Patrones y Aprendizaje Automatizado.

Proyecto 4: Sistema de Reconocimiento de canciones por género.

## Objetivo

**Clasificación de audio por género.** Dado un conjunto de clips de audio, el objetivo es construir una red convolucional que aprenda a distinguir los géneros al aprender de los espectrogramas de los audios.

## Dataset

El dataset es el [GTZAN Genre Collection](http://marsyasweb.appspot.com/download/data_sets/), una colección de 1000 clips de 30 segundos de canciones divididas en 10 géneros distintos (100 clips por género).

De estos clips se obtieron los espectrogramas, que resultaron en imágenes de 127 (muestras de la frecuencia) por 2900 (intervalos de tiempo discretizados) pixeles.

### Estructura

La estructura del dataset es la siguiente:

```
genres
  |
  +-> blues
  |     |
  |     +-> resized000.png
  |     |
  |     +-> resized001.png
  |     |
  |     +-> ...
  |
  +-> classical
  |     |
  |     +-> ...
  |
  +-> ...
```

### Descarga

El dataset se puede descargar del siguiente link:  [genres.tar.gz](https://drive.google.com/file/d/1H5wI0P9Q8NbobbLyGycc8lVtW5D_GBUe/view?usp=sharing "genres.tar.gz")

### Cargar el dataset

El modelo se puede cargar con la función `load_dataset()` del módulo `data.py`.

### Disclaimer

Los clips de música fueron ocupados en este proyecto con fines puramente académicos.

## Implementación

Las redes convolucionales se construyen en Python sobre [Keras](https://keras.io "Keras Documentation") con [TensorFlow](https://www.tensorflow.org/ "TensorFlow") como backend, por lo que estos paquetes son necesarios.

También es necesario el paquete `h5py` para guardar y cargar modelos ya entrenados, así como `Matplotlib`, `SciPy`, `NumPy` y `Scikit-Learn` para procesamiento de imágenes, tensores, etc.

## Ejecución

El modelo (clase) `Ensemble` permite construir clasificadores colectivos con redes convolucionales que tienen la misma arquitectura y está listas para trabajar con el dataset de espectrogramas.

### Construcción de la red

Puede construir un clasificador con N redes convolucionales como sigue:

```python
Ensemble(n_clasificadores = N, path = 'clasificadores')
```

donde `clasificadores` es el nombre del directorio donde se guardarán las redes que conforman el clasificador colectiivo una vez que hayan sido entrenadas en formato `.h5`.

### Entrenamiento

Para entrenar un clasificador colectivo use el método `fit()` proporcionando
el conjunto de entrenamiento, para la implementación usamos _bagging_ por lo
que al entrenar se generarán muestras aleatorias con reemplazo para cada red.

```python
clasificador.fit (X_train, y_train, # conjunto de entrenamiento y etiquetas
                  (X_test, y_test), # conjunto de evaluación y etqiquetas
                  bootstrap_percent = .9, # tamaño de la muestra aleatoria de X_train
                  batch_size = 128, # tamaño de los lotes a usar para entrenar
                  epochs = 15) # número de épocasa entrenar
```

### Probando la red

Una vez que tiene las redes entrenadas puede clasificar con los métodos `predict()` y `predict_classifiers()`

```python
# Las clases para las muestras en X
clases = clasificador.predict (X)

# v_clases son los vectores de predicción y l_clases las clases para las muestras en X
# Son las predicciones individuales de cada red
# Ver documentación para más información
v_clases, l_clases = clasificador.predict_classifiers (X)
```

### Cargando un modelo

Si ya tiene las redes entrenadas que componene un clasificador colectivo (archivos `.h5`), puede cargar el modelo de nuevo con el método `load_model()`.

```python
clasificador = Ensemble.load_model('clasificadores')
```

donde `clasificadores` es el directorio donde se encuentran los modelos entrenados.

Puede descargar un modelo ya entrenado (del que hablamos en el reporte) [aquí](https://drive.google.com/file/d/1bVLZ95FAZh-Fpc1m0l3-0YdQ4lSL9NB6/view?usp=sharing "clasificadores2.tar.gz").

### Nota

Se recomienda ampliamente el uso de una tarjeta gráfica NVIDIA para realizar los cómputos más rápido, esto si desea modificar los parámetros y/o agregar más capas al modelo.
