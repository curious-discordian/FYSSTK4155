# -*- coding:utf-8 -*-
## I'll be writing in Python 2 here.
## So if there's an issue, please check compabilities.
## (there's a few changes from Python 2 to 3, amongst them the removal
## of some good functional support, like reduce. :D)

## Implementation from the Mehta notebook. 
## I'll provide additional comments where I deem necessary.
## First off, let's make some changes to how the code is structured
## so that we can easily discern what is in the global scope, and
## what is not,
##
## A note on setattr + module :
## Using the module we can use setattr(module, variable-name, variable-value)
##
## Important note: using the setattr here will result in it becoming a global
## variable. (That is, we can use it internally in a funciton and set
## it outside the scope) 
##
## For the rest therefore, we can manipulate the internal workings a bit more
## freely. 

### -------------------- Imports ----------------------------------- ### 
import numpy as np
import scipy.sparse as sp
import scipy.signal as sig


np.random.seed(12)
from sys import modules
import sys
module = modules[__name__] # This one is a bit magical, see above.

from os import getcwd
current_dir = getcwd() + '/'

from sklearn.preprocessing import normalize # need this 
from sklearn import linear_model

import warnings
#Comment this to turn on warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

### ---------------- Parameters of ISING ------------------------------
# system size
L=40
N = 100000 #states

states=np.random.choice([-1, 1], size=(N,L))

def ising_energies(states,L):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    Note: The way this is set is that we create a filter kernel (-ish) which 
    Essentially is an offset identity-matrix. (so a rotation matrix, of sorts) 
    that is multiplied with -1. 
    What this will do is make neighbors effect eachother. (short and simple) 
    """
    J=np.zeros((L,L),)
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)

    return E

# calculate Ising energies
energies=ising_energies(states,L)



### ----- Linear agression. --------------------------------
## For time-constraints we'll save this for last.
## 





### ------------------- Phases, and 2D lattices- ----------------------
## First we need a pickle.
## Make sure the data is in a subfolder ./IsingData/Ising2DFM_reSample_L40_T=0.25.pkl
## NOTE: using repickled data, for python 2 compability. There is a certain bug in the
## pickling which took some figuring out. See the repickle function for how to. 
import pickle
def read_t(t,root=current_dir+"IsingData/"):
    data = pickle.load(open(root+'Ising2DFM_rePickled_L40_T=%.2f.pkl'%t,'rb'))
    data= np.unpackbits(data).astype(int).reshape(-1,1600)
    data[np.where(data==0)]=-1
    return data

def FFT_analysis(t,show_demo=False):
    """
    Pass a temperature in the valid data-pool. 
    optionally show_demo [boolean] to show an example. 
    Returns the percentage of samples at that point where the order
    is over 80%. 
    (Seemed like a fair place to start) 
    """
    def FFT_phase(data):
        # Assuming the data is singular here.
        analysis = np.abs(np.fft.fftshift(np.fft.fft2(data.reshape(L,L))))
        # Note the shift-function, which transposes q1 with q3 and q2 with q4.
        # This is a pretty standard way of dealing with FFT2, as the
        # positive frequencies are on the first half, and the negative on the second.
        analysis = normalize(analysis)
        return analysis

    data_read = read_t(t) 
    FFT = np.array([FFT_phase(x.reshape(L,L)) for x in data_read])
    # Now for the hypothesis; it seems like the data is ordered
    # when the max value is over about .8,
    # and unordered if it is not.
    # So let's take the max of each of these, and then 
    maxes = np.array([np.max(x) for x in FFT])
    percentage = np.sum(np.where(maxes>.80,1,0))/float(len(maxes))
    if show_demo:
        # Should ensure that the pick is representative of the
        # genereral cases (i.e. pick the mean max-value) 
        pick = np.argsort(maxes)[len(maxes)//2] #index of median value
        data = data_read[pick]
        plt.figure()
        plt.subplot(211)
        plt.contourf(data.reshape(L,L))
        xx,yy = np.meshgrid(np.linspace(-L/2,L/2,L),np.linspace(-L/2,L/2,L))
        plt.subplot(212)
        plt.contourf(xx,yy,FFT[pick])

        plt.show()
        
    return percentage





### ----------------------- Throwaway neural network: -----------------------------
## For my own sanity I'll implement this with some functional principles as well.
## Let's keep things simple, and short:


ReLu = lambda x: np.maximum(x,0)
D_ReLu = lambda x: np.where(x<=0,0,1)

## If we wish to simply try the Neural Network from the lectures; 
#from neural_network import NeuralNetwork
#NN = NeuralNetwork(two_states, two_states)
#NN.train()











if __name__ == "__main__":
    #coupling estimate: 

    
    ## Phase estimate; 
    test_t = 2.25
    testFFT = FFT_analysis(test_t,True)
    print "FFT-analysis estimates probability %.2f percent the data at temperature T=%s "%(testFFT,test_t)

    # Nerual Network coupling estimates:


    # Neural network Phase estimate: 


"""
## SKRATCH : 
#This is usable as feed-forward with some modifications (this does not currently 
# take into account the biases. 
def feedforward(activation, weights):
    # Does all in one fell swoop using functional programming
    f = activation #activation_function
    initialized = [np.array(self.input)] + self.weights
    reduce(lambda z,w: f(np.dot(z,w)), initialized)
def display(NeuralNetwork):
    ## Let's make it so we're able to see the updating as it happens:
    def __init__(self): 
    fig = plt.figure()
    
    plt.subplot(221) # depends on the layers.
    plt.contourf(self.weights1)
    plt.title('W1')
    
    plt.subplot(222)
    plt.contourf(self.layer1)
    plt.title('L1')
    
    plt.subplot(223)
    plt.contourf(self.weights2)
    plt.title('W2')
    
    plt.subplot(224)
    plt.contourf(self.layer2)
    plt.title('L2')

"""
