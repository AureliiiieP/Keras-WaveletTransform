import scipy.io
import numpy as np
from keras.utils import to_categorical, Sequence
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Activation, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD, RMSprop
from keras.datasets import cifar100
from models.DWT import DWT_Pooling


def WaveletCNN(input_size = (224,224,3), nb_classes=120):

    inputs = Input(shape = input_size)

    output = Conv2D(32, (3,3), padding="same")(inputs)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)
    output = Conv2D(32, (3,3), padding="same")(output)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)

    output = DWT_Pooling()(output)
    output = Conv2D(32, (3,3), padding="same")(output)
    output = Dropout(0.25)(output)

    output = Conv2D(64, (3,3), padding="same")(output)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)
    output = Conv2D(64, (3,3), padding="same")(output)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)

    output = DWT_Pooling()(output)
    output = Conv2D(64, (3,3), padding="same")(output)
    output = Flatten()(output)
    output = Dropout(0.25)(output)

    output = Dense(512, activation="relu")(output)
    output = Dropout(0.5)(output)

    output = Dense(nb_classes, activation="softmax")(output)

    model = Model(input = inputs, output = output)
    
    return model


filepath = 'weights/class_WaveletCNN.h5'
nb_classes = 100
batch_size = 64
epochs = 400
lr = 0.01
trainFactor = 0.8
imageShape = (32,32,3)

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

### Create and Train the Model
model = WaveletCNN(imageShape, nb_classes)
model.summary()
optimizer = SGD()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])


model.fit(
        x_train,
        y_train,
        validation_split=1-trainFactor,
        epochs=epochs, 
        verbose=2, 
        callbacks = [
                    ModelCheckpoint("temp.h5", monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
                    EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
            ], 
        )

model.load_weights("temp.h5")
model.save(filepath)
print("Training done")


### Model Evaluation
result = model.evaluate(x_test, y_test)
print("Wavelet")
print(result)



