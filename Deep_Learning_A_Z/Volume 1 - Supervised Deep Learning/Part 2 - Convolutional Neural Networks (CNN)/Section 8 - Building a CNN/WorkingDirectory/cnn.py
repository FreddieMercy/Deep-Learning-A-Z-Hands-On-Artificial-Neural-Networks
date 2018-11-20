# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

#Convolution
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3), activation='relu'))

#Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Convolution
classifier.add(Conv2D(32,(3,3), activation='relu'))

#Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(activation="relu", units=128))
classifier.add(Dense(activation="sigmoid", units=1))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

classifier.fit_generator(ImageDataGenerator(   rescale = 1./255,
                                               shear_range = 0.2,
                                               zoom_range = 0.2,
                                               horizontal_flip = True).flow_from_directory(  'dataset/training_set',
                                                                                             target_size = (64, 64),
                                                                                             batch_size = 32,
                                                                                             class_mode = 'binary'),
                         steps_per_epoch = 8000,#training_set how many pics
                         epochs = 25,
                         validation_data = ImageDataGenerator(rescale = 1./255).flow_from_directory('dataset/test_set',
                                                                                                    target_size = (64, 64),
                                                                                                    batch_size = 32,
                                                                                                    class_mode = 'binary'),
                         #test_set how many pics
                         validation_steps = 2000)

