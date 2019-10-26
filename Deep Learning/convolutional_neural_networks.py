import os

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Additional Convolution Layer
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_data_gen = ImageDataGenerator(rescale=1. / 255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_data_gen = ImageDataGenerator(rescale=1. / 255)

training_set = train_data_gen.flow_from_directory('dataset/training_set',
                                                  target_size=(64, 64),
                                                  batch_size=32,
                                                  class_mode='binary')

test_set = train_data_gen.flow_from_directory('dataset/test_set',
                                              target_size=(64, 64),
                                              batch_size=32,
                                              class_mode='binary')

classifier.fit_generator(training_set,
                         use_multiprocessing=False,
                         workers=1,
                         # verbose=2,
                         steps_per_epoch=8000,
                         # samples_per_epoch=8000 // 32,
                         # samples_per_epoch=8000,
                         # epochs=90,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000
                         # validation_steps=2000 // 32
                         # validation_steps=2000
                         )

# <Long Process>
# classifier.fit_generator(training_set,
#                          samples_per_epoch=8000,
#                          nb_epoch=25,
#                          validation_data=test_set,
#                          nb_val_samples=2000)