# Keras-WaveletTransform
This is an implementation of Wavelet Transform layer for denoising and classification from the paper 
"Multi-level Wavelet Convolutional Neural Networks" by Pengju Liu, Hongzhi Zhang, Wei Lian, and Wangmeng Zuo
The paper can be found at : https://arxiv.org/abs/1907.03128



## Denoising Application 
Trained on DIV2K_train and tested on DIV2K_valid, CBSD68, Set12 and Urban100
Launch init.py with parameters 
* -tr : Train dataset path (Train only) 
* -t : Test dataset path 
* -n : Noise level
* -s : Sliding window step (Test only)
* -lw : Weight path of the model to load (Test only)
* -a : Architecture of the model
* -m : Mode (Train or Test)
	
Example for training a model based on Unet with Noise level 15:
```
python3 init.py -tr DIV2K/DIV2K_train_HR/ -t DIV2K/DIV2K_valid_HR/ -n 15 -a unet -m train
```
Example for testing a pretrained model based on Wavelet with Noise level 50 and Sliding window step of 50p :
```
python3 init.py -t DIV2K/DIV2K_valid_HR/ -n 50 -a wavelet -m test -lw weights/DenoisingWavelet_50.h5 -s 50
```


## Classification Application
Train and test a new model on Cifar100:
Launch classificationCifar.py or classificationCifarWavelet.py



## Benchmark
For Denoising Application only.
Compares pretrained unet-based model to wavelet-based model following noise level (15, 25, 50).
Computes mean SSIM and mean PSNR over a dataset.
dataset_path and models in benchmark_list in the script can be changed for custom benchmark.


