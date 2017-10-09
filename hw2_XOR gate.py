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
time=1

class Layer(object):
     def __init__(self,inputX,inputSize,outputSize,activation_function=None):
         self.W=T.matrix()
         self.X=inputX
         self.B=T.vector()
         self.W=theano.shared(np.random.normal(0,1,(inputSize,outputSize)))
         self.B=theano.shared(np.zeros(outputSize)+0.1)
         print("init")
         print(self.W.get_value())
         #(self.B.get_value())
         #(inputX)
         self.WX_B=T.dot(self.X,self.W)+self.B
         #print(self.WX_B.eval())
         self.activation_function=activation_function
         if activation_function is None:
            self.outputs = self.WX_B
         else:
            self.outputs = self.activation_function(self.WX_B)
            
         #print(self.outputs.eval())
         
     def updateRule(self,layerIndex,layerLength,y): #Cr對W跟B微分 =Cr/Wij & =Cr/Bij
         print("update")
         
         cost =T.sum((self.outputs-y)**2) #cost function    
         if layerIndex==layerLength:
             print("output Layer")
             #   compute delta_L = a*sigma'(z)*c(y)
             C_grad=T.grad(cost,self.outputs)
             print(C_grad.eval())
             sigma_grad=self.WX_B*(1-self.WX_B)
             print(sigma_grad.eval())
             dw=T.matrix()
             print(self.X.eval())
             dw=self.X*T.dot(sigma_grad,C_grad)
             
             
         else:
             print("hidden layer")
             # x *sigma'(z)*(w) delta_(l-1)  
         
         
                     

for t in range(time):
     i=random.randrange(0,4)
     trainingX=T.vector()
     trainingX=X[i]
     trainingY=Y[i]
     #print(trainingX)
     #print("1")
     l1=Layer(trainingX,2,2,T.nnet.sigmoid)
     #l1.updateRule(1,2,trainingY)
     print(l1.outputs.eval())
     l2=Layer(l1.outputs,2,1,T.nnet.sigmoid)
    # print(l2.W)
     l2.updateRule(2,2,trainingY)
    # print(l2.W)
     
     
     