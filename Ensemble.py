#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.utils import resample
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import load_model

class Ensemble (object):
  """Clase Ensemble para implementación de clasificadores colectivos usando
  el módulo Keras y TensorFlow como backend.
  """

  def __init__(self, n_clasificadores = None, path = 'clasificadores'):
    """Crea un nuevo ensemble con el número de clasificadores dados.

    Parametros:
    -----------
    n_clasificadores : int, default: None
      El numero de clasificadores. Si es None el ensemble está vacío.
    path : str
      El directorio donde se guardarán los modelos entrenados que componen
      es clasificador colectivo self.

    Notas:
    ------
    Los clasificadores son homogéneos, esto es, todos tienen la misma
    arquitectura.
    """
    if n_clasificadores:
      self.path = path
      self.n_clasificadores = n_clasificadores
      self.ensemble = []
      for i in range(n_clasificadores):
        classifier = Sequential()

        classifier.add(Conv2D(filters = 100,
                              kernel_size = (127,15),
                              strides = (1,5),
                              input_shape = (127, 2900, 1),
                              activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (1,2)))

        classifier.add(Dropout(.5))

        #classifier.add(Conv2D(filters = 64,
        #                        kernel_size = (1,15),
        #                        strides = (1,5),
        #                        activation = 'relu'))
        #classifier.add(MaxPooling2D(pool_size = (1,5)))
        #
        #classifier.add(Dropout(.5))

        classifier.add(Flatten())

        classifier.add(Dense(128, activation = 'relu'))

        classifier.add(Dropout(.3))

        classifier.add(Dense(10, activation = 'softmax',
                             kernel_regularizer = l2(0.0001)))

        classifier.compile(optimizer = Adam(0.0001),
                           loss = 'categorical_crossentropy',
                           metrics = ['accuracy'])

        self.ensemble.append(classifier)

  def fit (self,X_train,y_train,validation_data,bootstrap_percent,
    batch_size = 32, epochs = 10):
    """Entrena y evalúa el clasificador colectivo con los datos de entrada.

    Parametros:
    -----------
    X_train,y_train : array
      Los datos para entrenamiento con sus respectivas etiquetas.
    validation_data : (array,array)
      Los datos de validación y sus respectivas etiquetas.
    bootstrap_percent : float
      El porcentaje del tamaño del conjunto de entrenamiento que será usado para
      entrenar cada clasificador.
    batch_size : int, default: 32
      El tamaño de los lotes que se usarán para el entrenamiento.
    epochs : int
      El número de épocas para el entrenamiento

    Regresa:
    --------
    self : Ensemble

    Notas:
    ------
    Los modelos que componen el clasificador colectivo se guardarán en un
    archivo .h5 una vez termine su entrenamiento.
    """
    train_size = X_train.shape[0]
    i = 0

    for classifier in self.ensemble:
      # Subconjunto de entrenamiento
      Xb,yb = bootstrap_set(X_train,y_train,
                            int(np.floor(train_size*bootstrap_percent)))
      
      print('\n# ===============================================================================')
      print('# Entrenamiento: Clasificador #%d' % i)
      print('# ===============================================================================\n')

      classifier.fit(Xb,yb,batch_size = batch_size,epochs = epochs,
                     validation_data = validation_data)
      
      classifier.save(os.path.join(self.path,'classifier_%d.h5' % i))
      
      i += 1

    return self

  def predict (X):
    """Calcula la clase a la que corresponden las muestras en el conjunto dado.

    Parametros:
    -----------
    X : array
      El conjunto de muestras.

    Regresa:
    --------
    v_cls : array [n_samples,n_classes]
      La lista con los vectores de clasificacion de cada muestra, de cada
      clasificador en el clasificador colectivo.
    l_cls : array [n_samples]
      La lista con las etiquetas de cada muestra, de cada clasificador en el
      clasificador colectivo.
    """
    v_cls = []
    l_cls = []

    for classifier in self.ensemble:
      L = classifier.predict (X)
      v_cls.append (L)
      l_cls.append(np.argmax(L,axis = 1))
    
    return v_cls,l_cls

  @staticmethod
  def load_model (path):
    """Carga un modelo ya entrenado.

    Parametros:
    -----------
    path : str
      El directorio donde se encuentras los modelos que componen el clasificador
      colectivo.

    Regresa:
    --------
    ensemble : Ensemble
      El modelo previamente entrenado.
    """
    l = []
    for root,_,files in os.walk(path):
      n_models = len(files)
      for file in files:
        path = os.path.join(root,file)
        l.append(load_model(path))
    
    ensemble = Ensemble(None)
    ensemble.path = root
    ensemble.n_clasificadores = n_models
    ensemble.ensemble = l
    
    return ensemble

  @staticmethod
  def load_dataset ():
    """Carga el dataset de imagenes de espectrogramas que estan en el directorio
    "genres/".

    Regresa:
    --------
    X : np.array
      Las imagenes de tamaño (127*2900).
    y : np.array
      Las etiquetas de las imagenes.
    """
    y = []
    X = []
    i = 0

    for root,_,files in os.walk('genres/'):
      if len(files) != 0:
        l = [0]*10
        l[i] = 1
        y = y + [l]*len(files)
        i += 1
      for file in files:
        path = os.path.join(root,file)
        img = plt.imread(path)
        if img.shape != (127,2900):
          img = misc.imresize(img,(127,2900))
        X.append(np.expand_dims(img,2))

    X = np.array(X)
    y = np.array(y)

    return X,y

  @staticmethod
  def bootstrap_set (S, labels, size, seed = None):
    """Genera una muestra aleatoria con remplazo de S del tamaño indicado.

    Parametros:
    -----------
    S : np.array
      El dataset de donde se extraera la muestra.
    labels : np.array
      Las etiquetas de los elementos de S.
    size : int
      El tamaño de la muestra.
    seed : int, default: None
      Semilla para el generador de numeros aleatorios de NumPy.

    Regresa:
    --------
    train_set : np.array
      La muestra aleatoria.
    test_labels : np.array
      Las etiquetas de los elementos de la muestra aleatoria
    """
    if seed:
      np.random.seed(seed)

    idx = resample(np.arange(S.shape[0]),replace = True,n_samples = size)

    train_set = S[idx]
    train_labels = labels[idx]

    return train_set,train_labels
