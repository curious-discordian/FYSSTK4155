
# -*- coding: utf-8 -*-
import sys  
#^mainly for LatexTemplate in other folder:
if sys.platform == 'darwin':
    sys.path.append('/Users/JacobAlexander/Dropbox/Py_Resources')
else:
    sys.path.append('/home/jacobalexander/Dropbox/Py_Resources/') 
# ^location of resources: LatexTemplate etc. 
from LatexTemplate import *
Type = Template(Title = "Project 3; Wildcard. ") #<----- Header title
#^additional vars: author(str), date(str), landscape(boolean) 
#---create content as raw string below--# 
#----------using LaTeX-format-----------#  
content = r'' #creating initial content, 
#------------content below--------------# 
content += r''' 

\begin{multicols*}{2}

\section*{Introduction}
In this project we are supposed to select our own data to make the methods 
we've covered in this course, and apply what we've learned. \\ 

In (typical by now) fashion, I've deviated from this. 

The concept that struck me was that it could be more interesting to analyse 
how we can use meta-programming to allow the computer to self-modify a 
neural network on any data input given only that it is told what the output should 
look like. \\ 

In order to experiment with this, we do need to go into the nuts and bones of what 
a neural network is, and also define some overarching principles for the reconfiguring 
methods. \\ 

Methodologically I've been a bit all over the place, so summarising it we will need to 
go back to the concepts from an atomic level. \\ 

Consider the following: 
\begin{itemize}
\item[] A network consists of layers 
\item[] A layer consists of Neurons 
\item[] A neuron's value is determined by it's activation function. 
\item[] The activation function is a kind of map-reduce. 
\end{itemize}
Specifically regarding the activation function: 
\begin{itemize}
\item[] mapping takes a series of values to another series of values. (one-to-one) 
\item[] reduction takes many values to a single value (many-to-one) 
\item[] The Reduction is currently done using matrix operations. 
\item[] The mapping function is usually just done by vectorization. 
\item[] There is no mixed strategies! 
\end{itemize}

Regarding the Linear Algebra: 
\begin{itemize}
\item[] Linear algebra is usually optimized for machines. 
\item[] Linear algebra is a way of sorting linear mappings. 
\item[] We use this because it's fast and easy. 
\item[] In physics, Linear Mappings, or Functionals, are a subset of the operations usually 
encountered 
\item[] The usage of chained linear mappings mixed with reductions create something like 
non-linear mappings. 
\item[] The non-linear combinations of elements follow in a linear fashion (usually increasing 
in order in parallel.) 
\item[] The "lower" order considerations (i.e. when the order of a $f(x,\dot{x};t)$ depends on 
the term $x(t)$) ) only occurs in situations where e.g. $\sigma(w_1 x_1 + w_2 x_2) | w_n = 0, where n \neq 1$, 
in other words; only when the weights of all other possible combinations are reduced to 0. 
\item[] Something which has solved this to some extent is ResNets, but that still only adds in the terms to the other inputs. 
\end{itemize}

Solutions: 
\begin{itemize}
\item[] Assume that DenseNets are somewhat helpful, but let's  further that by allowing the 
inputs to each layer to be specified differently. i.e. not just repeated LA, but also addition 
of "lower" terms. 
\item[] Allow the net to modify connections directly, and not just depend on slow gradient 
descent towards nilling out the weights that are not applicable. 
\item[] Also allow for re-interpretation of what a filter is (for conv-nets) 
\end{itemize}

Final key realisation (what is a kernel/filter) : 
\begin{itemize}
\item[] A filter takes a region of data, and outputs a single value. 
\item[] The way we use them they typically take inputs along a pattern. 
\item[] That pattern goes something like 3x3 matrix then skip one, then take another. 
\item[] Effectively these are a sum of a correlation-matrix. 
\item[] (In other words another reduction-type scheme, but here with a modifiable correlation-matrix)
\item[] These can then be "connected" together as smaller filters, (we should know this if 
we've read up on the Cooley-Tukey FFT, which utilizes Split Radix, something I think we glazed 
over in an earlier project) 
\end{itemize}

O.K, so that's a lot of listing. \\ 
 

\subsection*{Final combination: }
Thinking over these concepts for a while, we can come to the conclusion that we only use 
linear algebra in the way we do because it has worked for many before, and it is fast 
on modern computers due to years of optimization (see BLAS, ATLAS, etc). \\ 

There's nothing less or more magical with that. So we have a challenge now; how can we 
take all of this into a common framework of our own, and moreover map all this into a larger 
kind of framework where we can allow metaprogramming-like operations to take place while 
the neural network is Training? \\ 

To make it all more interesting, once we have the concepts locked into a superclass, what 
are the ideal ways of allowing this meta-programming to take place? \\ 

In the longer run, we'll include this in a different algorithm, where we view the layers 
as a kind of sentence, and a network as a paragraph. \\ 

It is not so far-fetched as it may sound to do this, but for the time being we won't do that. 
Our targets should take a bit more of a physics-based approach, and include states, entropy, and 
free energy. (We'll get back to this). \\ 

By now this may seem like a useless mess of concepts thrown throughout. 
A sort of crock-pot concoction, with little thought to the implementation. 
Hopefully by the time you've looked through the code, the amount of thought that went into 
this may be a little clearer. \\ 

Summarising the target again: \\
We just want our network to grow to fit the problem, without needless bloating, and 
be somewhat accurate at the end. \\ 
The intermittent states being allowed to grow, we need to subject each layer  
to some constraints, and limit the state-space using tools we should know from before. \\ 
Considering that information is describable using Entropy (Thanks to Shannon), we should 
feel free to explore whether other concepts that we physicists associate with our own 
Entropy apply here as well.\\ 
So we think of the neuron-states as configurations in a higher-dimensional phase-space.\\ 
By this logic, we can imagine that for $N=3$ what we get is a quadrant of a three-dimensional 
Ball. In this configuration we consider how a space in this ball maps to the next, etc., until 
we finally have a place in a final configuraton that represents 


\section*{The Cost/Loss duality. }



\end{multicols*}


'''  


#------useful functions & such----------#
# Allowing for Python Code
# %s format for standard string inclusion. 
# May also call on other programs for generating data 
# that goes into string. 

# See for table generation: 
# Use Table(Input, Input titles) function for Table 
#                          or Table.T for transpose
#Latex: 
'''

#%\end{Verbatim}
#%\VerbatimInput{program}

\mathcal{} #%Calligraphic
\mathbb{}  #Hollow Spaces
\mathbf{}  #Boldface

\begin{equiation}\tag*{S} #change showing tag *no parenthesis
\labe
\begin{Figure}
 \centering
 \includegraphics[width=\linewidth]{count_deg_fig.png}
 \captionof{figure}{målingsdata}
\end{Figure}

\begin{multicols*}{2} %remember to \end.
\begin{description} %or item, enumerate
\epigraph{quote}{author} %for cool quotes

\includepdf{file.pdf} %or png works too. 

% For "skipping" the rest of column, or page 
\vfill\null
\columnbreak

\begin{align} 
%usage: & for line \\ for breaks


'''













#--------------Typeset below:-------------------->#
if __name__ == '__main__':
    try:
        Type.content(content)       # may be irrelevant
        Type.Typeset('project3',showpdf=False) # <-filename
    except: 
        pass



