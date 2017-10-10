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
     def __init__(self,inputSize,outputSize):
         self.W=T.matrix()
         self.B=self.delta_L=self.delta_l=T.vector()
         self.W=theano.shared(np.random.normal(0,1,(inputSize,outputSize)))
         self.B=theano.shared(np.random.random_sample(outputSize))
         self.delta_L=theano.shared(np.zeros(outputSize))
         self.delta_l=theano.shared(np.zeros(inputSize)) #最後一層才會用到(L-1)
         #print("init")
         print("init_B",self.B.eval())
         print("init_W",self.W.eval())
        
         
     def updateRule(self,layerIndex,layerLength,delta_ll): #Cr對W跟B微分 =Cr/Wij & =Cr/Bij
         #print("update")
         if layerIndex==layerLength:
             #print("output Layer")
             #compute delta_L = a*sigma'(z)*c(y)
             print("WX_B",self.WX_B.eval())
             C_grad=T.grad(self.cost,self.outputs)  #
             print("C微分",C_grad.eval())
             sigma_grad=self.WX_B*(1-self.WX_B)     #
             print("sigma微分",sigma_grad.eval()) 
             self.delta_L=T.mul(sigma_grad,C_grad)  #
             print("微分相乘",self.delta_L.eval())
             
             dw=self.X*self.delta_L            #print(dw.eval())
             db=self.delta_L
             print(db.eval())
             print("Warnig 2",self.B.eval())
             #update new W,B
             self.W=self.W-(eta*dw)       # print(self.W.eval())
             self.B=self.B-(eta*db)   #print(self.B.eval())
            
             # compute delta_L-1=sigma'(z)*(w)*delta_L 
             self.delta_l=self.delta_L*self.W*sigma_grad   #print(self.delta_l.eval())
             print("GGGGG:",self.delta_l.eval())             
         else:
             #print("hidden layer")
             #compute delta_l= sigma'(z)*(w)*delta_(l+1)  
             sigma_grad=self.WX_B*(1-self.WX_B)      #print(sigma_grad.eval())
             self.delta_l=T.dot(sigma_grad,delta_ll) #print(self.delta_l.eval())   
             #compute Cr/Wij=a*delta_l  
             
             print("Warnig 1",self.B.eval())
             print(self.X)
             dw=self.delta_l[0]*self.X #print(self.dw.eval())
             db=self.delta_l[0]       #print(self.db.eval())
             #print(db.eval())
             #print(self.B.eval())
             #compute new W,B
             self.W=self.W-(eta*dw)   #print(self.W.eval())
             self.B=self.B-(eta*db)   #print(self.B.eval())
             
         #print("__________layerIndex___________",layerIndex)
         #print("update W",self.W.eval())
         print("update B",self.B.eval())
         
     def act(self,inputX,inputY,activation_function=None):
         self.X=inputX
         self.Y=inputY
         self.WX_B=T.dot(self.X,self.W)+self.B #print(self.WX_B.eval())
         #activtion function
         self.activation_function=activation_function
         if activation_function is None:
            self.outputs = self.WX_B
         else:
            self.outputs = self.activation_function(self.WX_B)#print(self.outputs.eval())
         #cost function   
         self.cost=T.sum((self.outputs-self.Y)**2)
         
         
         
         
# training data
X=[[0,0],[0,1],[1,0],[1,1]]
Y=[0,1,1,0]
eta=1
time=10


for t in range(time):
     i=random.randrange(0,4)
     trainingX=X[i]
     trainingY=Y[i]
     if t==0:
         print("start")
         l1=Layer(2,2)
         l1.act(trainingX,trainingY,T.nnet.sigmoid)
         #print("layer 1 cost",l1.cost.eval())
         l2=Layer(2,1)
         l2.act(l1.cost,trainingY,T.nnet.sigmoid)
         #print("layer 2 cost",l2.cost.eval())
         #backprogation update W,B
         l2.updateRule(2,2,0)
         l1.updateRule(1,2,l2.delta_l)
     else:
         l1.act(trainingX,trainingY,T.nnet.sigmoid)
         l2.act(l1.cost,trainingY,T.nnet.sigmoid)
         l2.updateRule(2,2,0)
         l1.updateRule(1,2,l2.delta_l)



print("__________layer  1___________")
print("Final W",l1.W.eval())
print("Final B",l1.B.eval())  
print("__________layer  2___________")
print("Final W",l2.W.eval())
print("Final B",l2.B.eval())     
print("cost",l2.cost.eval())
#predict data
    


      
     
     
     