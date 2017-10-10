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

class Layer(object):
     def __init__(self,inputX,inputSize,outputSize,activation_function=None):
         self.W=T.matrix()
         self.X=inputX
         self.B=self.delta_L=self.delta_l=T.vector()
         self.W=theano.shared(np.random.normal(0,1,(inputSize,outputSize)))
         self.B=theano.shared(np.zeros(outputSize)+0.1)
         self.delta_L=theano.shared(np.zeros(outputSize))
         self.delta_l=theano.shared(np.zeros(inputSize))
         #print("init")
         #print(self.W.get_value())
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
         
     def updateRule(self,layerIndex,layerLength,y,delta_l1): #Cr對W跟B微分 =Cr/Wij & =Cr/Bij
         #print("update")
         cost =T.sum((self.outputs-y)**2) #cost function    
         if layerIndex==layerLength:
             #print("output Layer")
             #compute delta_L = a*sigma'(z)*c(y)
             C_grad=T.grad(cost,self.outputs)       #print(C_grad.eval())
             sigma_grad=self.WX_B*(1-self.WX_B)     #print(sigma_grad.eval()) 
             self.delta_L=T.dot(sigma_grad,C_grad)  #print(self.delta_L.eval())
             self.dw=self.X*self.delta_L            #print(dw.eval())
             self.db=self.delta_L
            
             #update new W,B
             tempW=self.W.T-self.dw
             self.W=tempW.T          #print(self.W.eval())
             self.B=self.B-self.db   #print(self.B.eval())
     
             # compute delta_L-1=sigma'(z)*(w)*delta_L 
             self.delta_l=self.delta_L*self.W*sigma_grad   #print(self.delta_l.eval())
            

         else:
             #print("hidden layer")
             #compute delta_l= sigma'(z)*(w)*delta_(l+1)  
             sigma_grad=self.WX_B*(1-self.WX_B)      #print(sigma_grad.eval())
             self.delta_l=T.dot(sigma_grad,delta_l1) #print(self.delta_l.eval())   
             #compute Cr/Wij=a*delta_l  
             self.dw=self.delta_l[0]*self.X #print(self.dw.eval())
             self.db=self.delta_l[0]        #print(self.db.eval())
                        
             #compute new W,B
             self.W=self.W-self.dw   #print(self.W.eval())
             self.B=self.B-self.db   #print(self.B.eval())
     
     def act(self,inputNewX,activation_function=None):
           #print("training")
           self.X=inputNewX
           self.WX_B=T.dot(self.X,self.W)+self.B 
           self.activation_function=activation_function
           if activation_function is None:
              self.outputs = self.WX_B
           else:
              self.outputs = self.activation_function(self.WX_B)
         
         
# training data
X=[[0,0],[0,1],[1,0],[1,1]]
Y=[0,1,1,0]
eta=0.1
time=200  
for t in range(time):
     i=random.randrange(0,4)
     trainingX=T.vector()
     trainingX=X[i]
     trainingY=Y[i]
     if t==0:
         print("start")
         l1=Layer(trainingX,2,2,T.nnet.sigmoid)
         #print(l1.outputs.eval())
         l2=Layer(l1.outputs,2,1,T.nnet.sigmoid)
         l2.updateRule(2,2,trainingY,0)
         l1.updateRule(1,2,trainingY,l2.delta_l)
     else:
         l1.act(trainingX,T.nnet.sigmoid)
         l2.act(l1.outputs,T.nnet.sigmoid)
         l2.updateRule(2,2,trainingY,0)
         l1.updateRule(1,2,trainingY,l2.delta_l)
     
        


#predict data      
l1.act([1,1],T.nnet.sigmoid)
l2.act(l1.outputs,T.nnet.sigmoid)
print(l2.outputs.eval())       
     
     
     