# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 10:10:07 2016

@author: AKononov
"""

import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
    

class RNN:
    
    def __init__(self, vocab_dim, hidden_dim=100, bptt_truncate=4):
         # Assign instance variables
         self.vocab_dim = vocab_dim
         self.hidden_dim = hidden_dim
         self.bptt_truncate = bptt_truncate
         # Randomly initialize the network parameters
         # st = tanh( U*xt + W*s(t-1) ) - memory
         # ot = softmax(V*st) - output
         # self.U = np.random.uniform(-np.sqrt(1./vocab_dim), np.sqrt(1./vocab_dim), (hidden_dim, vocab_dim))
         # self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (vocab_dim, hidden_dim))
         # self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
         self.U = np.random.uniform(-1, 1, (hidden_dim, vocab_dim))
         self.V = np.random.uniform(-1, 1, (vocab_dim, hidden_dim))
         self.W = np.random.uniform(-1, 1, (hidden_dim, hidden_dim))
        
    def forward(self, x):
        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim))
        # For each time step...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward(x)
        return np.argmax(o, axis=1)
    


