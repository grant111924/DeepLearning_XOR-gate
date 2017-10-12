# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 23:21:37 2017

@author: DELL
"""

import theano
import theano.tensor as T
import random
import numpy as np
import matplotlib.pyplot as plt

def backpropagation(W,WX_B,delta):
         delta_l=delta
         #print(WX_B.eval())
         sigma_grad=WX_B*(1-WX_B)      #  
         #print("sigma微分",sigma_grad.eval()) 
         delta_l=T.dot(W.T,sigma_grad)  #
         #print("delta",delta_l.eval())
         return delta_l
         
def feedforward(inputX,W,B,inputY):
        WX_B=T.dot(inputX,W)+B #
        #print("WX_B",WX_B.eval())
        #activtion function
        outputs =1/(1+T.exp(-WX_B)) #
        #print(outputs.eval())
        #cost function   
        cost=T.sum((outputs-inputY)**2)
        return cost,outputs
    
def Delta_L(cost,sigmoid,y_hat):
        sigmoid_grad=(1-sigmoid)*sigmoid
        print(sigmoid_grad.eval())
        #_gra=cost-y_hat
        C_grad=T.grad(cost,sigmoid)
        print(C_grad.eval())
        delta=T.dot(sigmoid_grad,C_grad)
        #
        print(delta.eval())
        return delta
def DWB(delta,inputs):
        dw=inputs*delta
        db=delta
        return dw,db
def predict(out_W,out_B,hidden_W1,hidden_B1,hidden_W2,hidden_B2,inputs):
         a1=T.dot(inputs,hidden_W1)+hidden_B1
         a2=T.dot(inputs,hidden_W1)+hidden_B1#
         A=[a1,a2]
         #print(A.eval())
         #print(out_W.eval())
         #print(out_B.eval())
         Y=T.dot(A,out_W)+out_B 
         #print(Y.eval())
         result =1/(1+T.exp(-Y))   
         return result
def DWB_new(cost,b,w):
         dw=T.grad(cost,w)
         db=T.grad(cost,b)
         return dw,db
x=T.vector() #data feature value
a=T.vector() #hidden feature value
y=T.scalar() #data labeled value 0 or 1
r=T.scalar() #predict y


hidden_W1=hidden_W2=out_W=theano.shared(np.array([random.random(),random.random()],dtype=np.float64))
hidden_B1=hidden_B2=out_B=theano.shared(0.)
eta=0.7  #Learning rate

a1,a1_sigmoid=feedforward(x,hidden_W1,hidden_B1,y)
a2,a2_sigmoid=feedforward(x,hidden_W2,hidden_B2,y)
r,r_sigmoid=feedforward([a1,a2],out_W,out_B,y)
out_dw,out_db=DWB_new(r,out_B,out_W)
hidden_dw1,hidden_db1=DWB_new(r,hidden_B1,hidden_W1)
hidden_dw2,hidden_db2=DWB_new(r,hidden_B2,hidden_W2)

#out_dw,out_db=DWB_new(r,out_W,out_B)
#hidden_dw,hidden_db=DWB_new(r,hidden_W,hidden_B)
grad=theano.function([x,y],updates=[(hidden_W1,hidden_W1-eta*hidden_dw1),
                                     (hidden_B1,hidden_B1-eta*hidden_db1),
                                     (hidden_W2,hidden_W2-eta*hidden_dw2),
                                     (hidden_B2,hidden_B2-eta*hidden_db2),
                                     (out_W,out_W-eta*out_dw),
                                     (out_B,out_B-eta*out_db)])
X=[[0,0],[1,1],[1,0],[0,1]]
Y=[0,0,1,1]

time=1000
for t in range(time):
     i=random.randrange(0,4)
     trainingX=T.vector()
     trainingX=X[i]
     trainingY=Y[i]
     grad(trainingX,trainingY)
     
     #a1,a1_sigmoid=feedforward(trainingX,hidden_W1,hidden_B1,trainingY)
     #a2,a2_sigmoid=feedforward(trainingX,hidden_W2,hidden_B2,trainingY)
     #r,r_sigmoid=feedforward([a1,a2],out_W,out_B,trainingY)
     #out_Delta=Delta_L(r,r_sigmoid,y)
        
     #hidden_Delta1=backpropagation(hidden_W1,a1_sigmoid,out_Delta)
     #hidden_Delta2=backpropagation(hidden_W2,a2_sigmoid,out_Delta)
        
     #out_dw,out_db=DWB_new(r,out_B,out_W)
     #hidden_dw1,hidden_db1=DWB_new(r,hidden_B1,hidden_W1)
     #hidden_dw2,hidden_db2=DWB_new(r,hidden_B2,hidden_W2)
     #print(t)
     #print("out_B",out_B.eval())
     #print("out_db",out_db.eval())
     #hidden_W1=hidden_W1-eta*hidden_dw1
     #hidden_B1=hidden_B1-eta*hidden_db1
     #hidden_W2=hidden_W2-eta*hidden_dw2
     #hidden_B2=hidden_B2-eta*hidden_db2
     #out_W=out_W-eta*out_dw
     #out_B=out_B-eta*out_db
      
     #print("out_B",out_B.eval())
     #print("out_W",out_W.eval())
     #print("out_dw",out_dw.eval())
     #print("out_db",out_db.eval())
        
        
     #print("hidden_W",hidden_W.eval())
     #print("hidden_B",hidden_B.eval())
     #print("hidden_dw",hidden_dw.eval())
     #print("hidden_db",hidden_db.eval())
     

for i in range(len(X)):
    inputs=T.vector()   
    inputs=X[i]
    print("predict ",X[i])
    print(predict(out_W,out_B,hidden_W1,hidden_B1,hidden_W2,hidden_B2,inputs).eval())
 