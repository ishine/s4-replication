# Efficiently Modeling Long Sequences with Structured State Spaces

## Introduction

S4 is a new class of models that are based on the control theory: State space models. 

 $$ \begin{aligned} x'(t) &= \boldsymbol{A}x(t) + \boldsymbol{B}u(t) \ y(t) &= \boldsymbol{C}x(t) + \boldsymbol{D}u(t) \end{aligned} $$ 

In the paper, they have mentioned a couple of times
```
S4 can remember all the history. 
```
In our understanding, specifically, we think S4 was able to use 
encode the every time stamp into a hidden state that is of 500 dimensions. 

S4 has two different views: recurrent one, and convolution one. In the training time,
S4 efficiently trained and parameterized through a CNN model, 
but in the inference time, S4 can switch a RNN mode since learned parameters are shared. Hence in the inference time,
we can use the hidden state that is of 500 dimensions to reconstruct the entire sequence before time step t. 

## Replication Tasks

### ListOps

### IMDB Review Classification

## S4 Implementation

### S4 architecture

### 

## Experimental Results


## Conclusion
