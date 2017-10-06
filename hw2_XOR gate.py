# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 19:03:49 2017

@author: DELL
"""

import theano
import theano.tensor as T
import random
import numpy as np
import matplotlib.pyplot as plt
# training data
X=[[0,0],[0,1],[1,0],[1,1]]
Y=[0,1,1,0]


x1=x2=T.vector()
a1=a2=y=T.scalar()
eta=0.1
time=100
def parameterWXB(weight,vector,bias):#comupte z=wX+b 
    neuron=T.dot(weight,vector)+bias  
    return neuron    
def sigmoid(z):# sigmoid activation function => [0,1] interval value 
    sigmo=1/(1+T.exp(-z)) 
    return  sigmo    
   
#Hidden No.1 Neurons w1,b1,z1,a1
w1=theano.shared(np.array([random.random(),random.random()],dtype=np.float64),'w1')
b1=theano.shared(0.)
z1=parameterWXB(w1,x1,b1)
a1=sigmoid(z1)  # sigmoid activation function
                      
#Hidden No.2 Neurons w2,b2,z2,a2
w2=theano.shared(np.array([random.random(),random.random()],dtype=np.float64),'w2')
b2=theano.shared(0.)
z2=parameterWXB(w2,x2,b2)
a2=sigmoid(z2)  # sigmoid activation function


#Output Neurons wO,bO,zO,y
wO=theano.shared(np.array([random.random(),random.random()],dtype=np.float64),'wO')
bO=theano.shared(0.)
aX=[a1,a2]
zO=parameterWXB(wO,aX,bO)
aF=sigmoid(zO)
cost =T.sum((aF-y)**2) #cost function  
dw,db=T.grad(cost,[wO,bO]) #gradient compute
gradient = theano.function([aX,y],updates=[(wO,wO-eta*dw),(bO,bO-eta*db)])# update function

for t in range(time):
     i=random.randrange(0,4)
     trainingX=X[i]
     trainingY=Y[i]
     gradient(trainingX,trainingY)