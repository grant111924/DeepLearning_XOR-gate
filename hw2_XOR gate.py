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
         self.delta_L=theano.shared(np.zeros(outputSize))#最後一層才會用到(L-1)
         self.delta_l=theano.shared(np.zeros(inputSize)) 
         
         print("init")
         print("init_B",self.B.eval())
         print("init_W",self.W.eval())        
     """向後更新"""   
     
     
     
     def updateRule(self,delta): #Cr對W跟B微分 =Cr/Wij & =Cr/Bij
         #print("update")
         self.delta_l=delta
         #print("hidden layer")
         #compute delta_l= sigma'(z)*(w)*delta_(l+1)    
         #compute delta_L-1=sigma'(z)*(w)*delta_L 
         sigma_grad=self.WX_B*(1-self.WX_B)      #  print("sigma微分",sigma_grad.eval()) 
         self.delta_l=T.dot(self.W.T,sigma_grad)  #print("delta",self.delta_l.eval())
         dw=self.X*self.delta_l # print(dw.eval())
         db=self.delta_l      #print(db.eval())
         
         #compute new W,B
         self.W=self.W-(eta*dw)   #
         print("後",self.W.eval())
         self.B=self.B-(eta*db)   #
         print(self.B.eval())
             
         #print("__________layerIndex___________")
         #print("update W",self.W.eval())
         #print("update B",self.B.eval())
     
     """向前傳輸"""
     def feedforward(self,inputX,inputY): 
         self.X=inputX
         self.Y=inputY
         self.WX_B=T.dot(self.X,self.W)+self.B #print(self.WX_B.eval())
         #activtion function
         self.outputs =1/(1+T.exp(-self.WX_B)) #print(self.outputs.eval())
         #cost function   
         self.cost=T.sum((self.outputs-self.Y)**2)
         
         
         
         
# training data
X=[[0,0],[0,1],[1,0],[1,1]]
Y=[0,1,1,0]
eta=0.1
time=10
outW=theano.shared(np.array([random.random(),random.random()],dtype=np.float64),'w')#weight inital value=[1,1]
outB=theano.shared(0.)  #bias inital value=0

for t in range(time):
     i=random.randrange(0,4)
     trainingX=X[i]
     trainingY=Y[i]
     if t==0:
         print("start")
         l1=Layer(2,2)
        
     
     l1.feedforward(trainingX,trainingY)
     outLinear=T.dot(outW,l1.outputs)+outB
     outSigmoid=1/(1+T.exp(-outLinear))
     outCost=T.sum((outSigmoid-trainingY)**2)
     print("output Layer result:",outSigmoid.eval())
     
     #compute delta_L             
     sigma_grad=outLinear*(1-outLinear)#sigma'(z_L)
     #print("sigma微分",sigma_grad.eval()) 
     #CC_grad=-(trainingY-outLinear)
     C_grad=T.grad(outCost,outLinear) #dCost_r(dy_r)
     #print(CC_grad.eval())
     #print("C微分",C_grad.eval())
     delta_L=T.mul(sigma_grad,C_grad)
     #print("微分相乘",delta_L.eval())
         
     outDw=trainingX*delta_L
     outDb=delta_L
     outW=outW-outDw
     outB=outB-outDb
     l1.updateRule(delta_L)



print("__________layer  1___________")
print("Final W",l1.W.eval())
print("Final B",l1.B.eval())  
print("__________layer  2___________")
print("Final W",outW.eval())
print("Final B",outB.eval())     
#print("cost",l2.cost.eval())
#predict data     