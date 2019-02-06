import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from pathlib import Path
from skimage import exposure

def promeni_minst(X):

    ret_val = np.empty([len(X), 28, 28])
    for i in range(len(X)):    
        # print('data shape: ' + str(data.shape) + ', data type: ' + str(type(data)))
        img = X[i]   

        img = (img).astype('uint8')
        img = exposure.rescale_intensity(img, out_range=(0, 255))  #TODO: proveri sta radi xD
        
        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if h > 7:
                img_crop = img[y : y + h, x : x + w]
                img_resized = cv2.resize(img_crop, (28, 28)) 
                ret_val[i] = img_resized
                break    

    return ret_val

def kreiraj_mrezu():
   seed = 7 # zakucati random zbog ponovljivosti
   np.random.seed(seed)

   (X_train, y_train), (X_test, y_test) = mnist.load_data()
   X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
   X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

   # normalize inputs from 0-255 to 0-1
   X_train = X_train / 255
   X_test = X_test / 255
   # one hot encode outputs
   y_train = np_utils.to_categorical(y_train)
   y_test = np_utils.to_categorical(y_test)
   num_classes = y_test.shape[1]

   #print('y length ' + str(len(y_train)))

   # build the model
   mreza = Sequential()
   mreza.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
   mreza.add(MaxPooling2D(pool_size=(2, 2)))
   mreza.add(Dropout(0.2))
   mreza.add(Flatten())
   mreza.add(Dense(128, activation='relu'))
   mreza.add(Dense(num_classes, activation='softmax'))
	# Compile model
   mreza.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
   
   return mreza

def getKonvoluciona():
    ime_fajla = 'tezine.hdf5'
    mreza = kreiraj_mrezu()
    tezine_fajl = Path(ime_fajla)

    if tezine_fajl.is_file():
        print("fajl postoji:")
        mreza.load_weights(ime_fajla)
        print('mreza ucitana..')
        return mreza
    else:
        print("fajl ne postoji")
        # Fit the model
        seed = 7 # zakucati random zbog ponovljivosti
        np.random.seed(seed)

        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = promeni_minst(X_train)
        X_test = promeni_minst(X_test)

        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

        

        # normalize inputs from 0-255 to 0-1
        X_train = X_train / 255
        X_test = X_test / 255
        # one hot encode outputs
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        #num_classes = y_test.shape[1]

        print('y length ' + str(len(y_train)))

        mreza.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
        # Final evaluation of the model
        scores = mreza.evaluate(X_test, y_test, verbose=0)
        print("CNN Error: %.2f%%" % (100-scores[1]*100))

        # Sacuvaj mrezu
        mreza.save_weights(ime_fajla, overwrite=True)

        return mreza

