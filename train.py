
from glob import glob
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
import numpy as np

from models.unet import unet
from models.unetWavelet import unetWavelet
from utils import CustomGenerator, psnr



def train(FLAGS):
        
        ### Parameters of the training
        # Can be freely changed 
        epochs = 400
        train_ratio = 0.8
        batch_size = 16
        cropShape = (192,192,3)
        early_stop = 10
        # The learning rate is decayed from 10e-4 to 10e-5 over 200 epochs.
        optimizer = Adam(lr=10e-4, beta_1=0.9, beta_2=0.999, epsilon=10e-8, decay=10e-3) 


        train_path = FLAGS.train_path
        valid_path = FLAGS.test_path
        noise_level = FLAGS.noise
        architecture = FLAGS.architecture


        if architecture == "unet":
                filepath = "weights/DenoisingUnet"+"_"+str(noise_level)+".h5"
                model = unet(cropShape)

        elif architecture == "wavelet":
                filepath = "weights/DenoisingWavelet"+"_"+str(noise_level)+".h5"
                model = unetWavelet(cropShape)
        
        else : 
                raise RuntimeError('Unkwown architecture, please choose from "unet" or "wavelet".')

        #model.summary()
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[psnr])


        ### Preparation of the dataset, building the generators
        train_files =[img_path for img_path in glob(train_path + '/*.png')]
        test_files =[img_path for img_path in glob(valid_path + '/*.png')]

        train_generator  = CustomGenerator(train_files[0:int(len(train_files)*train_ratio)], noise_level, batch_size, cropShape)
        valid_generator  = CustomGenerator(train_files[int(len(train_files)*train_ratio):], noise_level, batch_size, cropShape)
        test_generator = CustomGenerator(test_files, noise_level, batch_size, cropShape)

        
        ### Train the model
        model.fit_generator(
                train_generator, 
                epochs=epochs, 
                verbose=2, 
                callbacks = [
                        ModelCheckpoint("temp.h5", monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
                        EarlyStopping(monitor='val_loss', patience=early_stop, verbose=0, mode='auto')
                ], 
                validation_data=valid_generator
                )
        model.load_weights("temp.h5")
        model.save(filepath)
        print("Training done")


        ### Evaluate the model on the test dataset
        result = model.evaluate_generator(test_generator)
        print(result)