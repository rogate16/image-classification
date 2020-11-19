from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense

# Setting Images Path
train_path = "kue-indonesia/train"
val_path = "kue-indonesia/validation"
test_path = "kue-indonesia/test"

# Set Callbacks
reduceLR = ReduceLROnPlateau(monitor="loss", patience=5, verbose=1, mode="min")
early_stop = EarlyStopping(monitor="loss", patience=5, verbose=1, mode="min",
                           restore_best_weights=True)

# Initiate Image Data Generator
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                    rotation_range=15,
                                    zoom_range=0.2,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    horizontal_flip=True,
                                    vertical_flip=True)

val_datagen = ImageDataGenerator(rescale=1.0/255,
                                    rotation_range=15,
                                    zoom_range=0.2,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    horizontal_flip=True,
                                    vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load Images to Generator
train_generator = train_datagen.flow_from_directory(train_path,
                                                   target_size=(200,200),
                                                   batch_size=16,
                                                   class_mode="categorical")

val_generator = val_datagen.flow_from_directory(val_path,
                                               target_size=(200,200),
                                               batch_size=16,
                                               class_mode="categorical")

test_generator = test_datagen.flow_from_directory(test_path,
                                               target_size=(200,200),
                                               batch_size=16,
                                               class_mode="categorical")

# Modelling
model = Sequential()

model.add(Conv2D(32, (3,3), activation="relu", input_shape=(200,200,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(8, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

model.fit_generator(train_generator, epochs=200, validation_data=val_generator,
                    callbacks=[reduceLR, early_stop])

model.save("model/model.h5", save_format=".h5")