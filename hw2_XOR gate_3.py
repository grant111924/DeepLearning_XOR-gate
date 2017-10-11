# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:41:37 2017

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
        #_gra=cost-y_hat
        C_grad=T.grad(cost,sigmoid)
        delta=T.dot(sigmoid_grad,C_grad)
        #print(delta.eval())
        return delta
def DWB(delta,inputs):
        dw=inputs*delta
        db=delta
        return dw,db
def predict(out_W,out_B,hidden_W,hidden_B,inputs):
         A=T.dot(inputs,hidden_W)+hidden_B #
         #print(A.eval())
         #print(out_W.eval())
         #print(out_B.eval())
         Y=T.dot(A,out_W)+out_B 
         #print(Y.eval())
         result =1/(1+T.exp(-Y))   
         return result
   
x=T.vector() #data feature value
a=T.vector() #hidden feature value
y=T.scalar() #data labeled value 0 or 1
r=T.scalar() #predict y
hidden_W=theano.shared(np.random.normal(0,1,(2,2)))
hidden_B=theano.shared(np.random.random_sample(2))
out_W=theano.shared(np.array([random.random(),random.random()],dtype=np.float64),'w')#weight inital value=[-1,1]
out_B=theano.shared(0.)  #bias inital value=0
eta=0.7  #Learning rate

a,a_sigmoid=feedforward(x,hidden_W,hidden_B,y)
r,r_sigmoid=feedforward(a,out_W,out_B,y)
out_Delta=Delta_L(r,r_sigmoid,y)
hidden_Delta=backpropagation(hidden_W,a_sigmoid,out_Delta)
out_dw,out_db=DWB(out_Delta,a)
hidden_dw,hidden_db=DWB(hidden_Delta,r)
grad=theano.function([x,y],updates=[(hidden_W,hidden_W-eta*hidden_dw),
                                     (hidden_B,hidden_B-eta*hidden_db),
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
     #print(t)
     #print("out_B",out_B.eval())
     #print("out_db",out_db.eval())
     #hidden_W=hidden_W-eta*hidden_dw
     #=hidden_B-eta*hidden_db
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
    print(predict(out_W,out_B,hidden_W,hidden_B,inputs).eval())
 