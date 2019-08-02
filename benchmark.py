from glob import glob
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
from models.DWT import DWT_Pooling, IWT_UpSampling
from utils import psnr
from skimage.measure import compare_psnr, compare_ssim


# Parameters of the benchmark, dataset_path and benchmark_list can be changed for custom dataset and models.
dataset_path = "Urban100/image_SRF_4/"
benchmark_list = [  (15, (("weights/DenoisingUnet_15.h5", {"psnr" : psnr}), ("weights/DenoisingWavelet_15.h5",{"psnr" : psnr, "DWT_Pooling" : DWT_Pooling, "IWT_UpSampling" : IWT_UpSampling}))),
                    (25, (("weights/DenoisingUnet_25.h5", {"psnr" : psnr}), ("weights/DenoisingWavelet_25.h5",{"psnr" : psnr, "DWT_Pooling" : DWT_Pooling, "IWT_UpSampling" : IWT_UpSampling}))),
                    (50, (("weights/DenoisingUnet_50.h5", {"psnr" : psnr}), ("weights/DenoisingWavelet_50.h5",{"psnr" : psnr, "DWT_Pooling" : DWT_Pooling, "IWT_UpSampling" : IWT_UpSampling}))),
                 ]

data_files =[img_path for img_path in glob(dataset_path + '*HR.png')]


results = []

for bench in benchmark_list :

    for model_path in bench[1] :
        model = load_model(model_path[0], custom_objects=model_path[1])
        cropShape = model.input.shape[1:]

        psnr = []
        ssim = []

        for img_path in data_files :
            image = img_to_array(load_img(img_path))
            ref_image = image[:(image.shape[0]//cropShape[0])*cropShape[0], :(image.shape[1]//cropShape[1])*cropShape[1]]
            noise = np.random.normal(0,bench[0],ref_image.shape)
            noisy_image = ref_image + noise
            noisy_image /= 255.0
            ref_image /= 255.0

            residual_image = np.empty(ref_image.shape)
            
            # Divide the image into subimages of size cropShape to match the model's input size.
            for block_y in range(0,image.shape[0]//cropShape[0]):
                extracts = []
                for block_x in range(0,image.shape[1]//cropShape[1]):
                    extracts.append(noisy_image[block_y*cropShape[0]:(block_y+1)*cropShape[0],block_x*cropShape[1]:(block_x+1)*cropShape[1]])

                extracts = model.predict(np.array(extracts))
                for block_x in range(0,image.shape[1]//cropShape[1]):
                    residual_image[block_y*cropShape[0]:(block_y+1)*cropShape[0],block_x*cropShape[1]:(block_x+1)*cropShape[1]] = extracts[block_x]
            
            corrected_img = noisy_image-residual_image
            psnr.append(compare_psnr(ref_image, corrected_img, data_range=1.0))
            ssim.append(compare_ssim(ref_image, corrected_img, data_range=1.0, multichannel=True))
        results.append((bench[0],model_path[0], np.mean(psnr), np.mean(ssim)))
print(dataset_path)
print(results)