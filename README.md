
# Machine Learning Adventures

Hello! This is a collection of experiments, re-implmentations, and fun things I have done in machine learning. All of the code will be written in Python and I commonly use the libraries numpy, keras, and pytorch

## Folders

Each of the folders contains the code for creating a deep learning model for that dataset. All the files should have comments explaning the overall thought process. Some folders may have pre-trained weights included. I plan on attatching accuracies and graphs later on.

## Current Folders

 - **kddcup1999** - A cybersecurity dataset which can be found [here](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
	 - Feed-forward neural network with categorical and continuous features
	  - Embedding layers utilized in PyTorch implementation and one-hot encoding used in Keras implementation
 - **wisdm** - A human activity dataset which can be found [here](http://www.cis.fordham.edu/wisdm/dataset.php)
	 - Recurrent neural network on time series [lstm]
	 - Implements sliding window to group data into trainable information
	 - Inspired by [Tomasz Bartkowiak's implementation](https://github.com/bartkowiaktomasz/har-wisdm-lstm-rnns)
## Useful resources
 - Overall
	 - [PyTorch Basics  - MorvanZhou](https://github.com/MorvanZhou/PyTorch-Tutorial)
	 - [PyTorch Basics  - yunjey](https://github.com/yunjey/pytorch-tutorial)
	 - [Overview of loss functions](https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7)
	 - [Save memory with pandas] (https://www.kaggle.com/marcmuc/large-csv-datasets-with-pandas-use-less-memory)
 - RNN (LSTM)
	 - [Basics](https://github.com/keras-team/keras/issues/2654)
	 - [LSTM Input](https://discuss.pytorch.org/t/understanding-lstm-input/31110)
	 - [Sliding window for time series](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/)
