# e4040-2020Fall-project
Valentine d'Hauteville vd2256  
Vishnusai Yoganand vy2163

Directory for the Columbia ECBME4040 Fall 2020 Final project.  
Our project investigates and reimplements results from the 2019 [paper](https://arxiv.org/abs/1905.11946) **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**


## Organization of this repo

This repository implements two main functionalities:

* Building an EfficientNet model from scratch and then validating its performance on an ImageNet test set by loading checkpoints weights into the model.
The code for building the model from scratch is in the utils.EfficientNet_model.py file and the notebook for evaluating its performance is eval_efficientnet_on_imagenet.ipynb. 
* Transfer training of a Keras EfficientNet-B0 model and MobileNet-V2 model on three open-source datasets: Cats vs Dogs, Standford Dogs and Cifar-10. 
The notebooks for running the evaluations on the three datasets are:
Transfer_Learning_Cats_vs_dogs.ipynb,
Transfer_Learning_Stanford_dogs.ipynb,
Transfer_Learning_CIFAR_10.ipynb
 
* Measure_FLOPS.ipynb compare the FLOPS between MobileNetV2 and EfficientNet-B0


## Datasets and models:

* eval_efficientnet_on_imagenet.ipynb used the ImageNetV2 dataset from this [repo](https://github.com/modestyachts/ImageNetV2). The weights are loaded into the model using checkpoints paths defined in the notebook.
* The transfer learning code makes use of the standard Cats vs Dogs, Stanford Dogs and CIFAR-10 made available to load directly, thanks to Tensorflow. 
* All the models for the transfer learning task are available [here](https://drive.google.com/drive/folders/1tbbA2RD7ube_oU99ia9MkagpzCXcNI6J?usp=sharing)

## How to run the code

The code can be run by running each notebook individually.

