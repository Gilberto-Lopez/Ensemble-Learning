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
#from keras.preprocessing.image import ImageDataGenerator

# =============================================================================
# Cargar datos
# =============================================================================

X,y = load_dataset()

train_size = int(np.floor(X.shape[0]*.7))
test_size = X.shape[0] - train_size

print('- Dataset: %d samples' % X.shape[0])
print('- Training set: %d samples' % train_size)
print('- Test set: %d samples' % test_size)

# Datos a usar
X_train,y_train = X[:train_size],y[:train_size]
X_test,y_test = X[train_size:],y[train_size:]

# =============================================================================
# Clasificador 1
# =============================================================================

classifier_1 = Sequential()

classifier_1.add(Conv2D(filters = 30,
                        kernel_size = (127,3),
                        strides = (1,2),
                        input_shape = (127, 2900, 1),
                        activation = 'relu'))
classifier_1.add(MaxPooling2D(pool_size = (1,2)))

classifier_1.add(Flatten())

classifier_1.add(Dense(64, activation = 'relu'))

classifier_1.add(Dense(10, activation = 'softmax'))

classifier_1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# =============================================================================
# Entrenamiento
# =============================================================================

Xb,yb = bootstrap_set(X_train,y_train,
                      int(np.floor(train_size*.8)))

classifier_1.fit(Xb,yb,batch_size = 32,epochs = 7,shuffle = False,
                 validation_data = (X_test,y_test))

classifier_1.save('classifier_1')