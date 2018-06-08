#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

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
