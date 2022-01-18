from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from matplotlib import pyplot as plt
import numpy as np

def train_detector(X_train, X_test, Y_train, Y_test, nb_filters = 32, batch_size=128, nb_epoch=5, nb_classes=2, do_augment=False, save_file='models/detector_model.hdf5'):
    # input image dimensions
    img_rows, img_cols = X_train.shape[1], X_train.shape[2]
    input_shape = (img_rows, img_cols, 3)

    X_train = X_train[:3000]
    X_test = X_test[:2000]
    rgb_X_train = np.repeat(X_train[..., np.newaxis], 3, -1)
    rgb_X_test = np.repeat(X_test[..., np.newaxis], 3, -1)
    #load pre trained model, exlcuding the last layer
    pre_trained_model = InceptionV3(input_shape = input_shape, include_top=False, weights='imagenet')

    for layer in pre_trained_model.layers:
        layer.trainable = False

    x = layers.Flatten()(pre_trained_model.output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(pre_trained_model.input, x)

    if nb_classes == 2:
        loss = 'binary_crossentropy'
        accuracy_plt = 'accuracy_detection.pdf'
        loss_plt = 'loss_detection.pdf'
    else:
        loss = 'categorical_crossentropy'
        accuracy_plt = 'accuracy_recognition.pdf'
        loss_plt = 'loss_recognition.pdf'

    model.compile(optimizer=Adam(lr=0.0001), loss=loss, metrics=['acc', 'loss'])

    history = model.fit(rgb_X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                        verbose=1, validation_data=(rgb_X_test, Y_test))
    score = model.evaluate(rgb_X_test, Y_test, verbose=0)
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(accuracy_plt)
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(loss_plt)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    model.save(save_file)
