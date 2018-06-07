#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 21:07:48 2018

@author: gilisaac
"""

import numpy as np
from data import load_dataset,bootstrap_set
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.regularizers import l2

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
# Clasificador 1
# =============================================================================

classifier_1 = Sequential()

classifier_1.add(Conv2D(filters = 100,
                        kernel_size = (127,15),
                        strides = (1,5),
                        input_shape = (127, 2900, 1),
                        activation = 'relu'))
classifier_1.add(MaxPooling2D(pool_size = (1,2)))

classifier_1.add(Dropout(.5))

#classifier_1.add(Conv2D(filters = 64,
#                        kernel_size = (1,15),
#                        strides = (1,5),
#                        activation = 'relu'))
#classifier_1.add(MaxPooling2D(pool_size = (1,5)))
#
#classifier_1.add(Dropout(.5))

classifier_1.add(Flatten())

classifier_1.add(Dense(128, activation = 'relu'))

classifier_1.add(Dropout(.2))

classifier_1.add(Dense(10, activation = 'softmax',
                       kernel_regularizer = l2(0.0001)))

classifier_1.compile(optimizer = Adam(0.0001),
                     loss = 'categorical_crossentropy',
                     metrics = ['accuracy'])

# =============================================================================
# Entrenamiento
# =============================================================================

# Bagging, cada modelo se entrena en una muestra con reemplazo del dataset de
# entrenamiento, se evalua con el conjunto de verificacion
Xb,yb = bootstrap_set(X_train,y_train,
                      int(np.floor(train_size*.9)))

classifier_1.fit(Xb,yb,batch_size = 32,epochs = 7,
                 validation_data = (X_test,y_test))

L = classifier_1.predict(X_test)
cls = np.argmax(L,axis = 1)

classifier_1.save('classifier_1.h5')