## Project description .

Short order; create tools to analyse the correlations in the Ising Model. 

The two main components of the Ising being the correlations active, and the 
strength of said correlations. 

Very short explanation of an Ising Model; 
Assume you have a binary question; yes/no

The simplest being a case of atoms in proximity of eachother. 

Assuming these have an effect upon eachother, in the ising model described in 
this project, the assumption is that one will be affected by its neighbor. 

# Energy. 

Physics introduce then a concept called Energy, which we define something like this: 
- Take each element of this model. 
- Compare all the neighbors. 
- Then count what the total difference or similarity is in this system. 

We say that since there is nothing else that these are compared to, that would 
influence this system, we decide to give this measurement a name, and we call it 
Energy. 

Nothing more or less than that. 


A simpler way; if we assume that all of these atoms push on each other, then 
we simply measure all the "push" that there is, and that is our Energy. 

# What is done to deal with this "push": 
As mentioned there are two things you can measure: 
What pushes on what (i.e. which things push the same, and which push opposite), 
and how hard do they push. 

If there is little or no push, the system is "ordered" (or if you will, if most 
things push in the same direction). 
If there is a lot of push in different directions, then the system is disordered. 

Secondly; we care how hard they push. 

# methods: 
The first bit is covered in the material, that is; to find the strength of 
each atoms effect. 

The second portion being to find the overall pattern of order/stability, i.e. the 
phase of the system (e.g. liquid or solid). This proved solvable using the FFT, to essentially 
classify the system based on its oscillations (or lack thereof). 

The main reason for that little experiment was due to the interesting graphics that it 
generates. 

# Second time around, the netowrks 
Finally there was the task to repeat these tasks using neural networks. 

Neural networks and multilayered perceptrons are good at lots of things, and so 
the task was to use these for the same regression and classification. 

Now a bit of simplification is always good, so when we determine how to solve things 
using programs, it's always good to keep in mind what you're trying to achieve. 

The first component needs to be the replication of the strength of the attraction, or 
repulsion inherit in each atom. 

The second is the configuration, correlation, which incidentally looks like one of 
the layers, or rather one of the kernels that you'd find in a convolutional neural 
network. 

So from that we can deal two birds with one hand in the bushes. 

Simple, and straight forward, with a little backpropogation, and with a little dab of 
heretical approach to how these neural networks "should work", or should be made to work. 

 

