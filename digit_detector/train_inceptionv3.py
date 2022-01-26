from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras as Keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator as ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np

def train_detector(X_train, X_test, Y_train, Y_test, nb_filters = 32, batch_size=32, nb_epoch=5, nb_classes=2, do_augment=False, save_file='models/detector_model.hdf5'):

   # X_train = X_train[:3000]
   # X_test = X_test[:2000]
    #rgb_X_train = np.repeat(X_train[..., np.newaxis], 3, -1)
    #rgb_X_test = np.repeat(X_test[..., np.newaxis], 3, -1)

    print('Adding resize layer')
    #resize images
    input_tensor = Keras.Input(shape=(32, 32, 3))
    input_tensor_resize = layers.Lambda(
        lambda image: Keras.backend.resize_images(
            image, (int(100 / 32)), (int(100 / 32)),
            "channels_last")
    )(input_tensor)

    print('Adding padding layer')
    # zero padding to get edges info, and a bigger picture
    y = layers.ZeroPadding2D(padding=4)(input_tensor_resize)

    #load pre trained model, exlcuding the last layer
    pre_trained_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=y)

    for layer in pre_trained_model.layers:
        layer.trainable = False

    x = layers.Flatten()(pre_trained_model.output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(pre_trained_model.input, x)

    if nb_classes == 2:
        loss = 'binary_crossentropy'
        accuracy_plt = 'v3_accuracy_detection.pdf'
        loss_plt = 'v3_loss_detection.pdf'
    else:
        loss = 'categorical_crossentropy'
        accuracy_plt = 'v3_accuracy_recognition.pdf'
        loss_plt = 'v3_loss_recognition.pdf'

    print('Compiling model')
    model.compile(optimizer=Adam(lr=0.0001), loss=loss, metrics=['acc', 'loss'])

    datagen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True,rotation_range=20,
                                 width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True,validation_split=0.2,
                                 preprocessing_function=gray_to_rgb)
    datagen.fit(X_train)

    train_it = datagen.flow(X_train, Y_train,batch_size=32, subset='training')
    val_it = datagen.flow(X_test, Y_test, batch_size=32, subset='validation')


    # confirm the iterator works
    batchX, batchy = train_it.next()
    print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

    print('Fit model')
    history = model.fit_generator(train_it,
                                  verbose=1,
                                  validation_data=val_it,
                                  validation_steps=len(X_test)/32,
                                  steps_per_epoch=len(X_train) / 32,
                                  epochs=2)
    score = model.evaluate_generator(datagen.flow(X_test, Y_test,batch_size=32),verbose=0, steps=X_test/32)
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


def gray_to_rgb(x):
    return np.repeat(x[..., np.newaxis], 3, -1)