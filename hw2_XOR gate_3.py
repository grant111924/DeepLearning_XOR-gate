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

def backpropagation(W,sigmoid,delta):
         #print(sigmoid.eval())
         sigma_grad=sigmoid*(1-sigmoid)       #print("sigma微分",sigma_grad.eval()) 
         delta_l=T.dot(W.T,sigma_grad)*delta  #print("delta",delta_l.eval())
         return delta_l
         
def feedforward(inputX,W,B,inputY):
        WX_B=T.dot(inputX,W)+B #print("WX_B",WX_B.eval())
        #activtion function
        outputs =1/(1+T.exp(-WX_B)) #print(outputs.eval())
        #cost function   
        cost=T.sum((outputs-inputY)**2)
        #cost = T.nnet.binary_crossentropy(outputs, y).mean()
        return cost,outputs
    
def Delta_L(sigmoid,y_hat):
        #cost function   
        #cost=T.sum((sigmoid-y_hat)**2)
        sigmoid_grad=sigmoid*(1-sigmoid) 
        #C_gra=cost-y_hat
        C_grad=sigmoid-y_hat
        delta=T.mul(sigmoid_grad,C_grad)
        #print(delta.eval())
        return delta
def DWB(delta,inputs):
        dw=inputs*delta
        db=delta
        return dw,db
def predic_old(out_W,out_B,hidden_W,hidden_B,inputs,y_hat):
         Z=T.dot(inputs,hidden_W)+hidden_B #
         A=1/(1+T.exp(-Z)) #print(outputs.eval())
         #cost function   
         Y=T.dot(A,out_W)+out_B 
         #print(Y.eval())
         result =1/(1+T.exp(-Y))  
         cost=T.sum((result-y_hat)**2)
         
         return cost
def DWB_new(cost,b,w):
         dw=T.grad(cost,w)
         db=T.grad(cost,b)
         return dw,db

    

x=T.vector() #data feature value
a=T.vector() #hidden feature value
y=T.scalar() #data labeled value 0 or 1
r=T.scalar() #predict y
hidden_W=theano.shared(np.random.normal(0,1,(2,2)))
hidden_B=theano.shared(np.array([1.,1.]))
out_W=theano.shared(np.array([random.random(),random.random()],dtype=np.float64),'w')#weight inital value=[-1,1]
out_B=theano.shared(1.)  #bias inital value=0
eta=0.7  #Learning rate

a_cost,a_sigmoid=feedforward(x,hidden_W,hidden_B,y)
r_cost,r_sigmoid=feedforward(a_sigmoid,out_W,out_B,y)
#out_Delta=Delta_L(r_sigmoid,y)
#hidden_Delta=backpropagation(hidden_W,a_sigmoid,out_Delta)
#out_dw,out_db=DWB(out_Delta,a_sigmoid)
#hidden_dw,hidden_db=DWB(hidden_Delta,x)
out_dw,out_db=DWB_new(r_cost,out_B,out_W)
hidden_dw,hidden_db=DWB_new(r_cost,hidden_B,hidden_W)
grad=theano.function([x,y],outputs=[r_cost],updates=[(hidden_W,hidden_W-eta*hidden_dw),
                                     (hidden_B,hidden_B-eta*hidden_db),
                                     (out_W,out_W-eta*out_dw),
                                     (out_B,out_B-eta*out_db)])
    
predict = theano.function(inputs=[x], outputs=[r_sigmoid])    
X=[[0,0],[1,1],[1,0],[0,1]]
Y=[0,0,1,1]

time=50000
cost_history = []

for t in range(time):
     i=random.randrange(0,4)
     trainingX=T.vector()
     trainingX=X[i]
     trainingY=Y[i]
     #if (i+1) % 5000 == 0:
        #print ("Iteration #%s: " % str(i+1))
       # print ("Cost: %s" % str(cost))
     cost=grad(trainingX,trainingY)
     cost_history.append(cost)
    
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
     
 #Plot training curve
plt.plot(range(1, len(cost_history)+1), cost_history)
plt.grid(True)
plt.xlim(1, len(cost_history))
plt.ylim(0, 1)
plt.title("Training Curve")
plt.xlabel("Iteration #")
plt.ylabel("Cost")
plt.show()
for i in range(len(X)):
    inputs=T.vector()   
    inputs=X[i]
    prediction=predict(inputs)
    print("predict:",prediction)
    #outputs=Y[i]
    #print("predict cost",X[i])
    #print(predict(out_W,out_B,hidden_W,hidden_B,inputs,Y[i]).eval())




