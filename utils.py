
from keras.utils import Sequence
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import keras.backend as K


class CustomGenerator(Sequence):

    """
    Custom Generator for Denoising Application :
    Returns pairs of noisy images and the corresponding residual images (the noise).

    # Arguments :
        img_path (String) : List of the path to the images of the dataset
        noise_level (int) : Variance of the Gaussian Noise to add to the images
        batch_size (int)
        cropShape (int, int, int) : Shape of the input image of the model
        shuffle (bool) : Shuffle image order at each epoch
    """

    def __init__(self, img_path, noise_level, batch_size, cropShape, shuffle=True):
        self.img_path = img_path
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.cropShape = cropShape
        self.indexes = np.arange(len(img_path))
        self.on_epoch_end()
        self.noise_level = noise_level
    
    def __len__(self):
        return int(np.floor(len(self.img_path) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        return self.__data_generation(indexes)

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):

        noisy_images = np.empty((self.batch_size, *self.cropShape), dtype=np.float32)
        noisy_residuals = np.empty((self.batch_size, *self.cropShape), dtype=np.float32)

        for i in range(self.batch_size) :
            image = load_img(self.img_path[self.indexes[i]])   
            image = img_to_array(image)         

            crop_index = np.array([np.random.randint(image.shape[0]-self.cropShape[0],size = 1)[0],np.random.randint(image.shape[1]-self.cropShape[1],size=1)[0]])

            image = image[crop_index[0]:crop_index[0]+self.cropShape[0], crop_index[1]:crop_index[1]+self.cropShape[1]]

            # Add Noise
            noise = np.random.normal(0,self.noise_level,self.cropShape)
            noisy_images[i] = image + noise
            noisy_residuals[i]= noise
            
        noisy_images=noisy_images.astype(np.float32)
        noisy_residuals=noisy_residuals.astype(np.float32)

        noisy_images /= 255.0
        noisy_residuals /= 255.0
        
        return noisy_images, noisy_residuals



def psnr(y_true, y_pred, max_value=1.0):
    # Keras metric for PSNR to compare two images. The higher, the better.
    mse = K.mean(K.square(y_pred - y_true), axis=-1)
    if mse == 0:
        return 100
    return 20.0 * K.log(max_value / K.sqrt(mse))/K.log(10.0)