import datetime
import tools as t
from sklearn.model_selection import train_test_split

# Keras imports
from keras.models import Sequential
from keras.layers import Activation, Convolution2D, Cropping2D, \
    Dense, Dropout, Flatten, Lambda, MaxPooling2D
from keras.callbacks import EarlyStopping

# Assemble list of available frames from data subdirectories
suffixes = ['{:0>2}'.format(n) for n in range(5, 18)]  # dirs 05 through 17
data_dir = 'data/'

data = t.assemble_data(suffixes, data_dir)
print('Available Samples: {}'.format(data.shape[0]))

# Set params and print out
ang_factor = 0.4
batch_size = 32
epochs = 20
base_samples = 11000  # samples grow 6x with data augmentation

print('Steering camera angle adjustment: {}'.format(ang_factor))
print('Batch size: {}'.format('batch_size'))
print('Epochs: {}'.format(epochs))
print('Samples to use (pre-data-augmentation): {}'.format(base_samples))

# Generate a random list of images to use from potential augmented images
df = t.random_image_list(data, ang_factor, base_samples)

# train test split
train, valid = train_test_split(df, test_size=0.2)

samples_per_epoch = len(train)
nb_val_samples = len(valid)

# generators to select and process data on the fly
train_generator = t.generator(train, batch_size=batch_size)
valid_generator = t.generator(valid, batch_size=batch_size)

# model definition
model = Sequential()
model.add(Cropping2D(cropping=((59, 20), (0, 0)), input_shape=(160, 320, 4)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(1, 1), border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, subsample=(1, 1), border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
model.add(Convolution2D(96, 3, 3, subsample=(1, 1), border_mode='valid'))
model.add(Convolution2D(110, 3, 3, subsample=(1, 1), border_mode='valid'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

# Human readable summary with output shapes
print(model.summary())

callbacks = [EarlyStopping(monitor='val_loss', patience=0, verbose=0)]

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    samples_per_epoch=samples_per_epoch,
                    nb_epoch=epochs,
                    callbacks=callbacks,
                    validation_data=valid_generator,
                    nb_val_samples=nb_val_samples
                    )

model.save('%s_v11_nvidia_angf_%s_%s.h5' % (train.shape[0], ang_factor, datetime.datetime.now()))
