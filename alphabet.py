import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
import rnn

# fix random seed for reproducibility
np.random.seed(42)

#data = open('input.txt', 'r').read() # should be simple plain text file
#alphabet = list(set(data))
#data_size, vocab_size = len(data), len(alphabet)
#print 'data has %d characters, %d unique.' % (data_size, vocab_size)

# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

# prepare the dataset of input to output pairs encoded as integers
seq_length = 1
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
	seq_in = alphabet[i:i + seq_length]
	seq_out = alphabet[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
	print(seq_in, '->', seq_out)
	
# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (len(dataX), seq_length, 1))
# normalize
X = X / float(len(alphabet))
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
print(dataX)
print(y)

vocabulary_size = len(char_to_int)

model = rnn.RNN(vocabulary_size)
o, s = model.forward(X[10])
print(o.shape)
print(o)