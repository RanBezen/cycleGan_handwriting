
# Generate handwriting using Deep learning
This project is a generative adversarial networks implementation for generate handwriting sentences. The project based on https://arxiv.org/abs/1703.10593 paper, and eriklindernoren project https://github.com/eriklindernoren/Keras-GAN  
In this project I implemented a Cycle Generative Adversarial Network, or short-  a "Cycle Gan". The Cycle Gan network receives two datasets and can produce an integrated picture of the two input images from the datasets. The datasets that I built for this task is printed sentences dataset and handwriting sentences dataset.

## The network
In order to get the best result, I tried several different architectures and compared them. All the models contain two parts: Generator and Discriminator.
The generator architecture is U-net. The U-net is convolutional network architecture for fast and precise segmentation of images. The network consists of Down sampling and Up sampling, when the number of filters increase in the first layers and decrease in the last layers.

## Datasets:
### Printed Dataset
The first dataset is printed sentences that I build from a pdf online book "Education and Philosophical Ideal By Horatio W Dresser". I converted the pdf into jpeg format, and I cut the lines. For be sure the lines will crop well, I cropped the printed frame area.
Then I resized the cropped image to the nearest integer that can divided by the rows number, and then, to achieve automatic line separator, it run with loop and cut the lines.
In order to avoid a blank line, the program throws lines whose number of bytes is less than a certain threshold.
for building the train/test datasets, I padded the images and resize them to the optimal size that I found the network working best- 512x48 
This dataset contain 614 grey scale 512x48 images for training, and 112 images for test.

### Handwriting Dataset
The second dataset is a handwriting sentences. This dataset has taken from "IAM Handwriting Database". The database contains forms of unconstrained handwritten text, which were scanned at a resolution of 300dpi and saved as PNG images with 256 gray levels.
In this project I used the lines of the forms data. In the data preparation I used several image processing method for cleaning the data and feet the size to 512x48 with the best quality
The method for cleaning the images was to remove high frequency contents from the images for inverse Fourier transform and threshold dropping. First I used Fourier Transform and inverse Fourier transform for removing low frequency components. Then the image with low frequencies removed from the original image. 
This dataset contain 578 grey scale 512x48 images for training, and 60 images for test.

## Network architectures
### Architecture 1:
#### Generator
The generator architecture is U-net with those layers:

Down sampling

+ Convolution 4x4 with 32 filters, strides=2
	+ Leaky-ReLu Activation function with 0.2 Alpha
	+ Instance Normalization
+ Convolution 4x4 with 64 filters, strides=2
	+ Leaky-ReLu Activation function with 0.2 Alpha
	+ Instance Normalization
+ Convolution 4x4 with 128 filters, strides=2
	+ Leaky-ReLu Activation function with 0.2 Alpha
	+ Instance Normalization
+ Convolution 4x4 with 256 filters, strides=2
	+ Leaky-ReLu Activation function with 0.2 Alpha
	+ Instance Normalization
	
Up sampling

+ Convolution 4x4 with 128 filters
	+ ReLu Activation function
	+ Instance normalization 
	+ Concat with the down sample output
+ Convolution 4x4 with 64 filters
	+ ReLu Activation function
	+ Instance normalization 
	+ Concat with the down sample output
+ Convolution 4x4 with 32 filters
	+ ReLu Activation function
	+ Instance normalization 
	+ Concat with the down sample output

#### Discriminator
+ Convolution 4x4 with 32 filters, strides=2
	+ Leaky-ReLu Activation function with 0.2 Alpha
+ Convolution  4x4 with 64 filters, strides=2
	+ Leaky-ReLu Activation function with 0.2 Alpha
+ Convolution  4x4 with 128 filters, strides=2
	+ Leaky-ReLu Activation function with 0.2 Alpha
+ Convolution  4x4 with 256 filters, strides=2
	+ Leaky-ReLu Activation function with 0.2 Alpha

### Architecture 2:
#### Generator
The generator architecture is U-net with those layers:

Down sampling

+ Convolution 3x3 with 32 filters, strides=2
	+ Leaky-ReLu Activation function with 0.2 Alpha
	+ Instance Normalization
+ Convolution 3x3 with 64 filters, strides=2
	+ Leaky-ReLu Activation function with 0.2 Alpha
	+ Instance Normalization
+ Convolution 3x3 with 128 filters, strides=2
	+ Leaky-ReLu Activation function with 0.2 Alpha
	+ Instance Normalization
+ Convolution 3x3 with 256 filters, strides=2
	+ Leaky-ReLu Activation function with 0.2 Alpha
	+ Instance Normalization

Up sampling

+ Convolution 3x3 with 128 filters, strides=1
+ Convolution 3x3 with 128 filters, strides=1
	+ ReLu Activation function
	+ Instance normalization 
	+ Concatenate with the down sample output
+ Convolution 3x3 with 64 filters, strides=1
+ Convolution 3x3 with 64 filters, strides=1
	+ ReLu Activation function
	+ Instance normalization 
	+ Concatenate with the down sample output
+ Convolution 3x3 with 32 filters, strides=1
+ Convolution 3x3 with 32 filters, strides=1
	+ ReLu Activation function
	+ Instance normalization 
	+ Concatenate with the down sample output

#### Discriminator

+ Convolution 3x3 with 32 filters, strides=2
	+ Leaky-ReLu Activation function with 0.2 Alpha
+ Convolution  3x3 with 64 filters, strides=2
	+ Leaky-ReLu Activation function with 0.2 Alpha
+ Convolution  3x3 with 128 filters, strides=2
	+ Leaky-ReLu Activation function with 0.2 Alpha
+ Convolution  3x3 with 256 filters, strides=2
	+ Leaky-ReLu Activation function with 0.2 Alpha

### Architecture 3:
#### Generator
The generator architecture is U-net with those layers:
Down sampling
+ Convolution 3x3 with 32 filters, strides=2
+ Convolution 3x3 with 32 filters
+ Convolution 3x3 with 32 filters
	+ Leaky-ReLu Activation function with 0.2 Alpha
	+ Instance Normalization
+ Convolution 3x3 with 64 filters, strides=2
+ Convolution 3x3 with 64 filters
+ Convolution 3x3 with 64 filters
	+ Leaky-ReLu Activation function with 0.2 Alpha
	+ Instance Normalization
+ Convolution 3x3 with 128 filters, strides=2
+ Convolution 3x3 with 128 filters
+ Convolution 3x3 with 128 filters
	+ Leaky-ReLu Activation function with 0.2 Alpha
	+ Instance Normalization
+ Convolution 3x3 with 256 filters, strides=2
+ Convolution 3x3 with 256 filters
+ Convolution 3x3 with 256 filters
	+ Leaky-ReLu Activation function with 0.2 Alpha
	+ Instance Normalization

Up sampling

+ Convolution 3x3 with 128 filters, strides=1
+ Convolution 3x3 with 128 filters
+ Convolution 3x3 with 128 filters
	+ ReLu Activation function
	+ Dropout 0.2 
	+ Instance normalization
	+ Concatenate with the down sample output
+ Convolution 3x3 with 64 filters, strides=1
+ Convolution 3x3 with 64 filters
+ Convolution 3x3 with 64 filters
	+ ReLu Activation function
	+ Dropout 0.2 
	+ Instance normalization 
	+ Concatenate with the down sample output
+ Convolution 3x3 with 32 filters, strides=1
+ Convolution 3x3 with 32 filters
+ Convolution 3x3 with 32 filters
	+ ReLu Activation function
	+ Dropout 0.2 
	+ Instance normalization 
	+ Concatenate with the down sample output

#### Discriminator
+ Convolution 3x3 with 32 filters, strides=2
+ Convolution 3x3 with 32 filters
	+ Leaky-ReLu Activation function with 0.2 Alpha
+ Convolution  3x3 with 64 filters, strides=2
+ Convolution  3x3 with 64 filters
	+ Leaky-ReLu Activation function with 0.2 Alpha
+ Convolution  3x3 with 128 filters, strides=2
+ Convolution  3x3 with 128 filters
	+ Leaky-ReLu Activation function with 0.2 Alpha
+ Convolution  3x3 with 256 filters, strides=2
+ Convolution  3x3 with 256 filters
	+ Leaky-ReLu Activation function with 0.2 Alpha

### Results
As we can see in the samples, the best results were of Architecture 2. It is noticeable that the results of Architecture 3 were over fitting.

#### Architecture 1
![Image description](https://github.com/RanBezen/cycleGan/blob/master/tmp/arch1.png)

#### Architecture 2
![Image description](https://github.com/RanBezen/cycleGan/blob/master/tmp/arch2.png)

#### Architecture 3
![Image description](https://github.com/RanBezen/cycleGan/blob/master/tmp/arch3.png)
