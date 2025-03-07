
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

import numpy as np


def train_detector(X_train, X_test, Y_train, Y_test, nb_filters = 32, batch_size=128, nb_epoch=5, nb_classes=2, do_augment=False, save_file='models/detector_model.hdf5'):
    """ vgg-like deep convolutional network """
    
    np.random.seed(1337)  # for reproducibility
      
    # input image dimensions
    img_rows, img_cols = X_train.shape[1], X_train.shape[2]
    
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3) 
    input_shape = (img_rows, img_cols, 1)
    #print(X_train.shape)
    #input_shape = X_train.shape

    model = Sequential()
    model.add(Conv2D(nb_filters, kernel_size,
                            padding='same', strides=(1,1),
                            input_shape=input_shape, data_format='channels_last'))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters, kernel_size, padding='same', strides=(1,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    print('1')
    model.summary()
    # (16, 8, 32)
     
    model.add(Conv2D(nb_filters*2, kernel_size, padding='same', strides=(1,1)))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters*2, kernel_size, padding='same', strides=(1,1),))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    print('2')
    model.summary()
    # (8, 4, 64) = (2048)

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    print('3')
    model.summary()

    accuracy_plt = ''
    loss_plt = ''
    if nb_classes == 2:
        loss = 'binary_crossentropy'
        #nb_epoch = 7
        accuracy_plt = 'accuracy_detection.pdf'
        loss_plt = 'loss_detection.pdf'
    else:
        loss = 'categorical_crossentropy'
        accuracy_plt = 'accuracy_recognition.pdf'
        loss_plt = 'loss_recognition.pdf'
    model.compile(loss=loss,
                  optimizer='adadelta',
                  metrics=['accuracy'])
    
    if do_augment:
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2)
        datagen.fit(X_train)
        history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            samples_per_epoch=len(X_train), epochs=nb_epoch,
                            validation_data=(X_test, Y_test))
    else:
        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(accuracy_plt)
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(loss_plt)
    plt.clf()
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    model.save(save_file)  


