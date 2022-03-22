# Efficiently Modeling Long Sequences with Structured State Spaces

## Introduction

S4 is a new class of models that are based on the control theory: State space models. 
```
x'(t) = Ax(t) + Bu(t)
y(t)  = Cx(t) + Du(t)
```

In this replication project, we are treating this state space model as established, and focused on 
*how can we  leverage deep neural networks to efficiently parameterize three matrixs A,B,C*.

**What S4 can do?**
In the paper, they have mentioned a couple of times:
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
