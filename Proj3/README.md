## Neural networks, and beyond.

This being the subcategory of what pertains to the culmination of material both 
within and outside the course FYSSTK4155, with some hopeful usefulness in a 
much longer run. 

The subdirectory of research and papers will just gently accrue some relevant 
research, while doc should contain the tex/pdf of the project of itself. 

The presentation will probably be somewhat haphazard, but I hope that in 
reading the documentation of the py-file for the neural network, once 
I've stripped it of fluff and ideation, will make sense to anyone who've been 
moderately subjected to a variety of mathematical fields. 

## On the project itself. 

The assignment of this project for the course was simply to apply to a freely 
chosen dataset the techniques that are taught in this course. 

As such I've elected to construct my own version of neural networks, building 
on what we know from optimisation, and once again removing us a step from the 
"plain" linear cases (that we do so adore, but which are ultimately toy-models) 


## The -meta- / abstract : 

In short, the missing elements of current neural networks are of the following: 

- non-linear effects, (i.e. in a moving image the next position of a two-lever 
pendulum will depend on the levers, and so be bound to a phase-space configuration, 
though that configuration may vary very wildly in some more chaotic cases) 

- Non-linear activation (allows compound statements) 

- terms propagating across layers beyond simply the next layer (or previous) 

- Convolutional filters are strictly speaking a form of correlation matrix. 
 (as such, why conform to the special case, instead of allowing arbitrary relations 
to arise in the process of identifying the underlying patterns) 

- A series of filters can under that intepretation both work to generalise, to 
find sprawling patterns, and in some cases act as blob-detectors (simply put, the 
filter can be stretched out or downsampled by applying another filter either by 
compositional rules for the kernel opertions, or by sequentially applying them.) 

- A "layer" is simply a way of sorting the neurons for ease of computation, but 
by that logic, we only need to have adaptive enough computations to handle these, and 
so there is no need to "lock" us in to the layered approach. 
By extension, we could look to the Inception net, which also evaluates loss (output-ish) 
at auxillary levels. This essentially means that we have multiple output levels, or 
in this framework; we can "learn" at multiple levels of abstraction. 
(If you read up on the Inception-cells the idea is somewhat here as well. where they 
concatenate compound filters.)





We also see similar concepts back with Residual Nets, and DenseNets: 
https://www.jeremyjordan.me/convnet-architectures



The logic of allowing some "first-order" terms to be included in the higher-levels 
should be fairly straight-forward to opine, though the specifics of neural networks 
that only allow for linear mappings neccesarily means that the length of the contributions 
to each neuron is essentially locked, (not accounting for drop-outs) 

This should be possible to reduce / deduce directly instead of having to rely on a limited 
channel for forward-propogation of the information. 

To take an example: an eye has a circular shape (conceptually a lower-level abstraction), that is 
nested in another circle (together forming a higher level relationship), which is also within 
folds of skin that generally conform to a kind of shape, most likely surrounded with shadows that 
would indicate that they are somewhat "pushed" inwards when compared to the surrounding structure 
of skull.


I'll ramble if I go on,  but ostensibly the concepts of a lower level abstraction combined with 
a higher-level abstraction are both contributing factors. 

Moreover the Filters used in convolution are more like correlation matrices. 



Since I started this approach, some of the concepts I've touched upon in the 
progress of writing this code seems to have made it into publication: 
https://www.technologyreview.com/s/612561/a-radical-new-neural-network-design-could-overcome-big-challenges-in-ai/

This appears to approximate the trajectories that characterizes the Lorenz' equations, a simple 
set of three variables in an ODE, with interactions across (i.e. one variable will depend 
on the derivatives of the other two, e.g. x[n + 1] = dx * (y + .8 z), or similar cases.) 
  
