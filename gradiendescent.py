import numpy as np
from wordset import *
import math
import itertools
from datetime import datetime

def train(nr_epochs=5, categories=['fiction'], vecLength = 10, negSampleSize=15, window_size=5,learning_rate=0.001):
    ws = WordSet(categories,vecLength,negSampleSize)
    for i in range(nr_epochs):
        train_epoch(ws,window_size,learning_rate)
    return ws

def train_epoch(ws,window_size,learning_rate):
    word_count = sum((map(len,ws.corpus)))
    print(word_count)
    acc_words = 0
    for text in ws.corpus:
        #list(map(lambda i: train_window(ws,get_window_slice(text,window_size,i), text[i],learning_rate),range(window_size,len(text)-window_size)))
        for i in range(window_size,len(text)-window_size):
            acc_words += 1
            dateTimeObj = datetime.now()
            done = 100.0*acc_words/word_count
            print(str(dateTimeObj) +": " + str(done)+"%" )
            center = text[i]
            window = text[i-window_size:i] + text[i+1:i+window_size+1]
            train_window(ws,window,center,learning_rate)

def get_window_slice(text,window_size,i):
    dateTimeObj = datetime.now()
    print(dateTimeObj)
    win1 = itertools.islice(text, i-window_size,i)
    win2 = itertools.islice(text, i+1, i+window_size+1)
    return list(itertools.chain(win1, win2))

def train_window(ws,window,center,learning_rate):
    samples = list(map(lambda w: ws.drawNegSample(w),window))
    grads = list(map(lambda pair: gradients(ws, pair[0], center, pair[1],learning_rate),zip(window,samples)))
    list(map(lambda grad: update_weights(ws,grad),grads))

def gradients(ws,outside,center,neg_sample,learning_rate):
    u = ws.uVector(outside)
    v = ws.vVector(center)
    u_matrix = ws.uMatrixSlice(neg_sample)
    return (outside, learning_rate*u), (center, learning_rate*v), (neg_sample, learning_rate*u_matrix)

def update_weights(ws,gradients):
    ws.updateUVector(*gradients[0])
    ws.updateVVector(*gradients[1])
    ws.updateUMatrixSlice(*gradients[2])

def gradient_v(v,u,u_matrix):
    grad_m = np.dot(u_matrix.transpose(),np.dot(u_matrix,v))
    grad_u = u*(sigma(np.dot(u,v))-1)
    return grad_m + grad_u

def gradient_u_matrix(v,u_matrix):
    v1 = vec_sigma(np.dot(u_matrix,v)).reshape(-1,1)
    v2 = v.reshape(1,-1)
    return np.dot(v1,v2)

def gradient_u(v,u):
    return v*(sigma(np.dot(u,v)-1))

def vec_sigma(x):
    return np.vectorize(sigma)(x)

def sigma(x):
    return 1/(1+math.exp(-x))

