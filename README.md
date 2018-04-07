# deep_learning
* /logistic_regression/: use logistic regression to identify whether or not a image includes a cat (75% accuracy)
* /shallow_neural_networks/: use shallow neural networks with in hidden layer to make planar data classification (90.25% accuracy with 50 hidden units)
* /deep_neural_networks/: use multi-layer neural networks to check whether or not there is a cat in the image (209 training examples, 50 test examples; 2-layer with 7 units achieves 72% accuracy; 4-layer with 32 units achieves 80% accuracy)
* /gradient_checking/: achieve the numerical gradient checking for a specific deep neural network (3-layers)
* /signs_recognition/: signs (about single nubmers) recognition: three-layer ConvNet (CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED), 1080 training examples, 120 test examples
* /residual_networks/: achieve the signs recognition again, using 50-layer residual networks by Keras under TensorFlow.
* /recurrent_neural_networks/: generate dinosaurs names by RNN
* /word_analogies/: use pre-trained word embedding matrix for word analogies (glove6b50d dataset)
* /emoji_prediction/: use LSTM and pre-trained word embedding for adding emoji for sentences. (Embedding -> LSTM -> Dropout -> LSTM -> Dense -> Softmax; word embedding from glove6b50d dataset)
