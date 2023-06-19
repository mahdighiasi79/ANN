# Handwritten-digit-recognition
This project aims to read handwritten digits with an Artificial Neural Network.
The well-known MNIST dataset has been used for this task. The dataset contains 60,000 train images and 10,000 test images.
Each image represents only a single digit.
At the first step, the code written in the "data_loader.py" file will fetch the data from the related ubyte files.
We also normalize the input data. So, all of the pixels in an image will have a quantity between 0 and 1.

The model we used for this task is a Feed Forward Neural Network with two hidden layers, with each layer having 16 neurons.
Because we are reading a single digit, we have 10 classes of images. Therefore, the output layer of our ANN has 10 neurons.
The network gets 784 inputs which is the number of pixels for a given image in the MNIST dataset.
The activation function of the hidden layers is ReLU. However, as we are doing a classification task, the activation function of the output layer is Softmax.

We have used the cross entropy loss function because it is suitable for classification tasks.
The algorithm used for optimization is Adam with hyperparameters as follow:
learning rate: 0.01
batch_size = 50
epochs = 10

After training the ANN, it will have an accuracy around 90% for classifying test images.

The project has been implemented solely by PyTorch.
