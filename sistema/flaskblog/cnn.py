from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import display
from PIL import Image
import numpy as np
from keras.preprocessing import image

def cnn(nombre):
    classifier = Sequential()

    classifier.add(Convolution2D(
        32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Flatten())

    classifier.add(Dense(output_dim=128, activation='relu'))

    classifier.add(Dense(output_dim=1, activation='sigmoid'))

    classifier.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory(
        'flaskblog/dataset/training_set',
        target_size=(64, 64),
        #bach_size= 32,
        class_mode='binary')

    test_set = test_datagen.flow_from_directory(
        'flaskblog/dataset/test_set',
        target_size=(64, 64),
        #bach_size=32,
        class_mode='binary')


    classifier.fit_generator(
        training_set,
        #steps_per_epoch=50,
        steps_per_epoch=5,
        epochs=3,
        validation_data=test_set,
        validation_steps=50)
    path = 'flaskblog/static/'+nombre
    test_image = image.load_img(path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    training_set.class_indices
    if result[0][0] >= 0.5:
        prediction = 's'
    else:
        prediction = 'c'
    return prediction
