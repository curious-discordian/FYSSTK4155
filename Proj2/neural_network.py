# -*- coding:utf-8 -*-
### General Stuff for Neural Networks,
## -



### -------------------- Imports ----------------------------------- ###
# General purpose inports, will be usable at later point 
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

######################################################################
# Neural Networks, Simplified:                                       #
#                                                                    #
# The Target here is to make a lightweight neural network model      #
# that is easy to write, and modifiable enough that one should be    #
# able to cover all the basics of how a neural network works.        #
#                                                                    #
# In order to do that, we may constitute it from a simple array of   #
# individual neurons, where there should be input (of a certain      #
# dimensionality), and output (of another dimensionality).           #
#                                                                    #
# If we can make the individual neurons, and adapt any forward or    #
# backward propogation through these, the motion of each constituent #
# neuron in the network should be plainly visible to a student,      #
# or a teacher, alike.                                               #
#                                                                    #
# With that in mind; let's decifer what goes into each and every     #
# neuron and try to get an intuition about it.                       #
#                                                                    #
######################################################################



#---------------------------Overview --------------------------------#
######################################################################
# Input(s):             Value:                  output(s):           #
#                                                                    #
#  x1 - - - w1 -                                                     #
#                \    [ w1 x1 + ]                                    #
#  x2 - - - w2 - -> F { w2 x2 + } + B  ====>   Activation            #
#                /    [ w3 x3   ]                                    #
#  x3 - - - w3 -                                                     #
#                    f(w.T dot x)               Some number.         #
#                                                                    #
#             z = w1 x1 + w2 x2 + w3 x3 + B                          #
#                                                                    #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
#              Connected 3 input -> 2 neurons.                       #
# -----------------------------------------------------------------  #
#                                                                    #
#  x1 - - - -                                                        #
#      \      \                                                      #
#        \     (N1) = F (w11 x1 + w21 x2 + w31 x3) + B1  => A1       #
#          \ //                                                      #
#  x2 - - - -                                                        #
#         /  \\                                                      #
#       /      (N2) = F (w12 x1 + w22 x2 + w32 x3) + B2  => A2       #
#     /       /                                                      #
#  x3 - - - -                                                        #
#                                                                    #
#                                                                    #
#                                                                    #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
#               Cost Function; standard:                             #
# -----------------------------------------------------------------  #
#                                                                    #
#   x1                                                               #
#      \            - > Cost = (A1 - expect)^2                       #
#        - - - A1 ={                                                 #
#      /            - > ---- = [(f(w11 x1 + w21 x2) + b1) - y]^2     #
#   x2                                                               #
#                                                                    #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
#                    Backpropogation:                                #
# ------------------------------------------------------------------ #
#                                                                    #
#                           | d C               |  d A               #
#   x1                      | --- =  2 (A - y)     --- = df(z)       #
#  dx1 --------             | d A               |  d z               #
#              \            |                                        #
#                - (N) -> C |                   |                    #
#              /            |  d z                d z                #
#   x2 --------             | ----- = x_n       | --- = 1            #
#  dx2                      | d w_n               d B                #
#                                                                    #
#                                                                    #
######################################################################
# INTERMISSION:                                                      #
# Key point: To generate a modifiable neural network, we will need   #
# to program some useful methods of extending it. These form mostly  #
# a basis for educational purposes, but with some creativity we will #
# make the concepts of neural networks dance for us.                 #
#                                                                    #
# As this is not intended as "just" an educational exercise, we can  #
# Implement some very interesting "novel" approaches.                #
#                                                                    #
# Firstly; the extensibility needs an adaptable framework to work    #
# upon. To create this we want to assign layers, and the actions     #
# between layers, as generative concepts. Think of what it _Should_  #
# be able to do, and not how it typically looks.                     #
#                                                                    #
# As a general rule we want the neurons to wire together after they  #
# fire together (this is now an "old" idiom). The typical way of     #
# allowing this to happen is essentially to attach everyone from the #
# start, and then implement dropout. (some connections go dark)      #
#                                                                    #
# Another potentially interesting point is the concept of allowing   #
# the neurons to "try" wiring to others.                             #
#                                                                    #
# Quick note on neurology; it seems that there is a limit of about   #
# nine outgoing connections for a neuron. Now since the case is that #
# we only deal with one-dimensional transfer, and not neurons that   #
# fire "backwards" in the chain, for now we can simulate somehting   #
# like it by pruning signals that drop too far.                      #
#                                                                    #
# To overcome some of the issues of this, we can also allow for the  #
# neurons to "test" new connections as the system is running.        #
# By essentially adding some random noise to each propogation.       #
#                                                                    #
# NOTE: This is not usually done, as far as I know.                  #
# Especially since we will be doing this on the weights.             #
#                                                                    #
# Another thing to take into account is the normalization of the     #
# incoming signal for each neuron.                                   #
#                                                                    #
# This is sort of done already, under the name of Weight-/Layer-     #
# Normalization. Logically this makes sense.                         #
#                                                                    #
# END INTERMISSION                                                   #
######################################################################
#                                                                    #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
#                   Extending neurons:                               #
# ------------------------------------------------------------------ #
#        LAYER 1  |  Activations in layer 1:                         #
# x1 ====--       |                                                  #
#        \ \      |  A1 =  w11 x1 + w21 x2 + w31 x3 + w41 x4         #
# x2 =======(N1)  |                                                  #
#         XX      |                                                  #
# x3 =======(N2)  |                                                  #
#        / /      |  A2 =  w12 x1 + w22 x2 + w32 x3 + w42 x4         #
# x4 ====--       |                                                  #
#                                                                    #
# NOTE:                                                              #
# We should notice now that this easily translates to matrices,      #
# but in order to allow us some flexibility in size we will make     #
# this matrix mutable.                                               #
# (this is built in to the numpy package, so we pass that parameter) #
#                                                                    #
#                                                                    #
#                                                                    #
#                                                                    #
######################################################################




"""
# The Basic outline: 
## For the case where we essentially are looking for how much neighbors are
## related, the simplest case is often the easiest.
##
## We want a simple type of neural network that can do two things, which is
## really one thing: It should discern the coupling coefficient of items
## in an array.
##
## What we are interested in then is what the pattern is, i.e. it should learn
## by iteratively altering the coupling matrix, and finally returning said matrix
## for us to see that it indeed finds the "neighbor" matrix. (this is what a
## different neural network would have as a hidden component)
##
## In some sense also we wish for this to work like an RNN, where it can continuously 
## feed back into itself, 
## 
## The simple case we wish to solve is dependent upon two things; 
## Since we know the target: 
## Firstly; it evaluates to a single value, which will be akin to the 
##          eigenvalue of a unit-matrix. So we need to allow for this. 
## 
## Secondly; the network needs to allow for a coefficient matrix to arise, 
##           which we shall make a secondary cost, as well as output layer. 
##
## This means we need to take what we know of neural networks, and weave it 
## together with what we know they can do to accomodate these. 
##
## In order to extract the correlations, there needs to be a layer of similar 
## size as the input. 
## That means that we need to borrow from three types of neural networks: 
## First: 
##      CNN - convolutional kernels are all about what the internal 
##            relationship is between values of a previous layer. (features) 
## Secondly: 
##      RNN - The concept of having multiple outputs at different levels of  
##            micro/macro scales is exactly what we see in LSTMs, 
## Thirdly: 
##  example - There was an example of a neural network that I wanted 
##            to point to for the argument of multiple outputs along a 
##            deep neural network, but I can't remember if it was Graves or deepmind
##  
## All of this put together, we can build around the goals. 
## We also have a few keys elements we know, as well as the target output size.
## 
## 
## --- Overarching view: ------------------------------------ 

The goal here is to essentially break the canonical way of constructing neural 
networks, and re-fashion it as an intuitive connection. 

Strictly speaking for the two outcomes; coupling constant, probability of ordered, 
comes down to a fairly simple estimate of derivatives. 

That means that if we're using a convolutional neural network it should end up with 
a Toeplitz of sorts. (or a kind of permutation matrix) 
In one case the filter-kernel should end up doing something similar to taking the 
gradient of the image, and then evaluating on that. 

Keywords 1 : 
Apply Circulant Toeplitz Matrix by using kernel =  [- x , _0_ , x ], 
Solve for x, and "learn" x == -1  

Keywords 2 : 
Recurring ||[Haar-wavelet -> downsample]|| => metric of change. 
Should yield a conceptual total gradient on multiple levels. 


The center functionality of this then should cover an algebraic ring (-ish). 
So; we will initialize with a simple array of random numbers. 
The array then describes a circulant matrix, and we evaluate Cost based on 
the Energy. 





"""
class SimpleNeurons:
    """
    Mixed case, some convolution, and some linear passes. 
    # Target 1:
    - Single output, target is the coefficient J 
    
    # Target 2: 
    - matrix/Kernel output, target is the coupling filter. 
    """
    
    def __init__(self, X,Y,f=None,df=None):
        width,depth = X.shape

        # Layers: Constraints; layer 1 must match input,
        #                      final layer must match output. 
        layer1 = np.random.random((width,depth))
        layer2 = np.random.random()
        layer3 = np.random.random()

        # Weights:
        # Constraints: must map from one layer to next.
        
        self.X = X
        self.Y = Y
        self.output = np.zeros(Y.shape)
        self.layers = layers

        # Dealing with the activation function: 
        if f and df:
            if callable(f) and callable(df):
                self.f = f
                self.df = df
            else:
                print "activation-function and derivative needs to be callable" 
        else:
            #Simple ReLu as default.  
            self.f = lambda x: np.maximum(x,0)
            self.df = lambda x: 1 if x > 0 else 0

        
    def feed_forward(self, f):
        # This will simply charge forward without storing the intermittent layers
        # anywhere: 
        # f ==  activation function
        # map,reduce ? -> map: applies a unary function to every element of an array.
        #                 red: applies a binary function recursively over every element.
        # Put together to create this:
        # 
        f = self.f
        initialized = [np.array(self.X)] + self.layers
        for i in range(len(self.layers)):
            setattr(self.layers[i],
                    f(layers,np.dot(initialized[i], )),
        self.output = reduce(lambda z,w: f(np.dot(z,w)), initialized)

    def convolutional_layer(self):
        # For ease of use, we will 
        # 
        pass
    
    def backprop(self,df):
        """
        Target: Make the easiest intuition on backpropogation, 
        I.e. modifiable and easy to implement. 
        """
        df = self.df
        # simplest way of sloping;
        
        pass
        

class Simple2d(SimpleNeurons):
    # Subclass of simple-neurons.
    #
    pass




## Or using the Neural Network example from lecture notes. 

class NeuralNetwork:
    ## From Lecture Notes, and modified to suit temperament.
    def __init__(self,
                 X_data,
                 Y_data,
                 # Default activation set to ReLu and deriv of ReLu 
                 activation_function=lambda x: np.maximum(x,0),
                 D_activation_function=lambda x: np.where(x<=0,0,1),
                 n_hidden_neurons=50,
                 n_categories=10,
                 epochs=10,
                 batch_size=100,
                 eta=0.1,
                 lmbd=0.0,):
        self.X_data_full = X_data
        self.Y_data_full = Y_data

        ## Supply activation function and derivative of
        ## activation functionn.
        self.activation = activation_function
        self.Dactivation = D_activation_function
        
        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):
        # feed-forward for training
        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = self.activation(self.z_h)

        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias

        exp_term = np.exp(self.z_o)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def feed_forward_out(self, X):
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = self.activation_function(z_h)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        
        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities

    def backpropagation(self):
        error_output = self.probabilities - self.Y_data
        deriv = self.Dactivation
        error_hidden = np.matmul(error_output, self.output_weights.T) * deriv(a_h)
        # self.a_h * (1 - self.a_h) # = derivative of sigmoid

        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()
