# MultiModalClassifier
This is a project repo for multi-modal deep learning classifier with popular models from Tensorflow and Pytorch. The goal of these baseline models is to provide a template to build on and can be a starting point for any new ideas, applications. If you want to learn basics of ML and DL, please refer this repo: https://github.com/lkk688/DeepDataMiningLearning.

# Bonus Work 2 by Devansh Modi

### Task 1 - Intel PyTorch Acceleration

- First step is to install the Intel Extension for PyTorch
  git clone --recursive https://github.com/intel/intel-extension-for-pytorch
  cd intel-extension-for-pytorch
  git checkout v1.11.0

  # if you are updating an existing checkout
  git submodule sync
  git submodule update --init --recursive

  python setup.py install
  
- Follow code changes were made for Intel Extension Acceleration

  import intel_extension_for_pytorch as ipex
  ...
  model = ipex.optimize(model)
  ...

Without any changes, the model is taking a long time to train.

Here is the first epoch iteration and the accuracy achieved after 20 minutes.

  The model has 23272266 trainable parameters
  Epoch 0/14
  ----------
  train Loss: 1.7236 Acc: 0.3295
  val Loss: 1.4562 Acc: 0.4534

  Epoch 1/14
  ----------
  train Loss: 1.3543 Acc: 0.5077
  val Loss: 1.2355 Acc: 0.5581

  Epoch 2/14
  ----------
  train Loss: 1.1905 Acc: 0.5802
  val Loss: 1.1523 Acc: 0.5958

  Epoch 3/14
  ----------
  train Loss: 1.0786 Acc: 0.6221
  val Loss: 1.1125 Acc: 0.6215

  Epoch 4/14
  ----------
  train Loss: 1.0011 Acc: 0.6514
  val Loss: 1.0101 Acc: 0.6541

  Epoch 5/14
  ----------
  train Loss: 0.9268 Acc: 0.6802
  val Loss: 1.0527 Acc: 0.6346

  Epoch 6/14
  ----------

- Now I will enable the Intel extension for PyTorch.


### Instructions for Setup

- Get flower data set using the dataset tools ~ `getflowertraintestdataset.py`
 i.e `https://s3.amazonaws.com/video.udacity-data.com/topher/2018/September/5baa60a0_flower-photos/flower-photos.zip`

- Install this project in development mode
```bash
  $ python setup.py develop


# Code organization
* [DatasetTools](./DatasetTools): common tools and code scripts for processing datasets
* [TFClassifier](./TFClassifier): Tensorflow-based classifier
  * [myTFDistributedTrainerv2.py](./TFClassifier/myTFDistributedTrainerv2.py): main training code
  * [myTFInference.py](./TFClassifier/myTFInference.py): main inference code
  * [exportTFlite.py](./TFClassifier/exportTFlite.py): convert form TF model to TFlite
* [TorchClassifier](./TorchClassifier): Pytorch-based classifier
  * [myTorchTrainer.py](./TorchClassifier/myTorchTrainer.py): Pytorch main training code
  * [myTorchEvaluator.py](./TorchClassifier/myTorchEvaluator.py): Pytorch model evaluation code 

# Tensorflow Lite
* Tensorflow lite guide [link](https://www.tensorflow.org/lite/guide)
* [exportTFlite](\TFClassifier\exportTFlite.py) file exports model to TFlite format.
  * testtfliteexport function exports the float format TFlite model
  * tflitequanexport function exports the TFlite model with post-training quantization, the model size can be reduced by
![image](https://user-images.githubusercontent.com/6676586/126202680-e2e53942-7951-418c-a461-99fd88d2c33e.png)
  * The converted quantized model won't be compatible with integer only devices (such as 8-bit microcontrollers) and accelerators (such as the Coral Edge TPU) because the input and output still remain float in order to have the same interface as the original float only model.
* To ensure compatibility with integer only devices (such as 8-bit microcontrollers) and accelerators (such as the Coral Edge TPU), we can enforce full integer quantization for all ops including the input and output, add the following code into function tflitequanintexport
```bash
converter_int8.inference_input_type = tf.int8  # or tf.uint8
converter_int8.inference_output_type = tf.int8  # or tf.uint8
```
  * The check of the floating model during inference will show false
```bash
floating_model = input_details[0]['dtype'] == np.float32
```
  * When preparing the image data for the int8 model, we need to conver the uint8 (0-255) image data to int8 (-128-127) via loadimageint function
  
# TensorRT inference
Check this [Colab](https://colab.research.google.com/drive/1aCbuLCWEuEpTVFDxA20xKPFW75FiZgK-?usp=sharing) (require SJSU google account) link to learn TensorRT inference for Tensorflow models.
Check these links for TensorRT inference for Pytorch models: 
* https://github.com/NVIDIA-AI-IOT/torch2trt
* https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/
* https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorrt/
