from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam

INIT_LR = 1e-3
EPOCHS = 50

datagen = ImageDataGenerator(rescale=1. / 255)

model = Sequential()
model.add(Conv2D(64, (2, 2),padding = 'same', input_shape=( 256, 256, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate = 0.5))

model.add(Conv2D(128, (2, 2),padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate = 0.2))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(11))
model.add(Activation('softmax'))

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss='categorical_crossentropy',
              optimizer = opt,
              metrics=['accuracy'])

batch_size = 32

train_generator = datagen.flow_from_directory("D:\\Work\\Progarmming\\Datasets\\data-release\\pytrain_grey\\test\\current", class_mode='categorical')
validation_generator = datagen.flow_from_directory("D:\\Work\\Progarmming\\Datasets\\data-release\\pytrain_grey\\valid\\current", class_mode='categorical')
model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=14,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
model.save('currentV2.h5')