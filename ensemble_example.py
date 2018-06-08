#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Ensemble import Ensemble
from data import load_dataset
import numpy as np

# =============================================================================
# Crear un modelo con 5 clasificadores, se guardan en el directorio
# clasificadores/, el directorio debe existir
# =============================================================================

Test = Ensemble(5,'clasificadores2')

# =============================================================================
# Cargar datos
# =============================================================================

# Dataset completo
X,y = load_dataset()

total_size = X.shape[0]
train_size = int(np.floor(total_size*.8))
test_size = total_size - train_size

print('- Dataset: %d samples' % total_size)
print('- Training set: %d samples' % train_size)
print('- Test set: %d samples' % test_size)

np.random.seed(54)
rp = np.random.permutation(total_size)
idx_train = rp[:train_size]
idx_test = rp[train_size:]

# Datos a usar para entrenamiento y verificacion
X_train,y_train = X[idx_train],y[idx_train]
X_test,y_test = X[idx_test],y[idx_test]

# =============================================================================
# Entrenamos el modelo
# =============================================================================

# Con una GPU GTX1060 este procedimiento toma ~1s por época
Test.fit (X_train,y_train,(X_test,y_test),
          bootstrap_percent = .9,
          batch_size = 128,
          epochs = 15)

# =============================================================================
# Cargar el modelo colectivo con componentes entrenadas en clasificadores/
# =============================================================================

Test = Ensemble.load_model('clasificadores')
