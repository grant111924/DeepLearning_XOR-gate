# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 13:47:49 2017

@author: DELL
"""

import theano
import theano.tensor as T
import random
import numpy as np
import matplotlib.pyplot as plt

def backpropagation(W,WX_B,delta):
         delta_l=delta
         sigma_grad=WX_B*(1-WX_B)      #  print("sigma微分",sigma_grad.eval()) 
         print("sigma_grad: ",sigma_grad.type)   
         delta_l=T.dot(W.T,sigma_grad)  #print("delta",self.delta_l.eval())
         print("delta_l: ",delta_l.type)   
         return delta_l
         
def feedforward(inputX,W,B,inputY):
        WX_B=T.dot(inputX,W)+B #print(self.WX_B.eval())
        #activtion function
        outputs =1/(1+T.exp(-WX_B)) #print(self.outputs.eval())
        #cost function   
        cost=T.sum((outputs-inputY)**2)
        return cost,outputs
    
def Delta_L(cost,sigmoid):
        sigmoid_grad=(1-sigmoid)*sigmoid
        print("sigmoid_grad: ",sigmoid_grad.type)            
        C_grad=T.grad(cost,sigmoid)
        print("C_grad: ",C_grad.type)
        delta=T.mul(sigmoid_grad,C_grad)
        print("delta: ",delta.type)
        return delta
def DWB(delta,inputs):
        dw=inputs*delta
        db=delta
        return dw,db
   
x=T.dmatrix() #data feature value
a=T.dvector() #hidden feature value
y=T.dscalar() #data labeled value 0 or 1
r=T.dscalar() #predict y
hidden_W=theano.shared(np.random.normal(0,1,(2,2)))
hidden_B=theano.shared(np.random.random_sample(2))
out_W=theano.shared(np.array([random.random(),random.random()],dtype=np.float64),'w')#weight inital value=[-1,1]
out_B=theano.shared(np.random.random_sample(1))  #bias inital value=0
eta=random.random()  #Learning rate
print("hidden_W   :",hidden_W.eval())
print("hidden_B   :",hidden_B.eval())
print("typeB",hidden_W.type)
print("typeW",hidden_B.type)
print("out_W   :",out_W.eval())
print("out_B   :",out_B.eval())

a,a_sigmoid=feedforward(x,hidden_W,hidden_B,y)
r,r_sigmoid=feedforward(a,out_W,out_B,y)
out_Delta=Delta_L(r,r_sigmoid)
hidden_Delta=backpropagation(hidden_W,a_sigmoid,out_Delta)
out_dw,out_db=DWB(out_Delta,a)
hidden_dw,hidden_db=DWB(hidden_Delta,r)
print("out_B",out_B.type)
print("out_W",out_W.type)
print("out_dw",out_dw.type)
print("out_db",out_db.type)


print("hidden_W",hidden_W.type)
print("hidden_B",hidden_B.type)
print("hidden_dw",hidden_dw.type)
print("hidden_db",hidden_db.type)
grad=theano.function([x,y],updates=[(hidden_W,hidden_W-eta*hidden_dw),(hidden_B,hidden_B-eta*hidden_db),(out_W,out_W-eta*out_dw),(out_B,out_B-eta*out_db)])


X=[[0,0],[1,1],[1,0],[0,1]]
Y=[0,1,0,0]

time=10
for t in range(time):
     i=random.randrange(0,4)
     trainingX=X[i]
     trainingY=Y[i]
     grad(trainingX,trainingY)

     
    