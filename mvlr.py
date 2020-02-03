import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

#function to compute Cost given X (input data) y (output data) and theta
def computeCost(X,y,theta):
    inner = np.power(((X @ theta.T)-y),2)
    return np.sum(inner)/(2 *len(X))

#function to perform gradient Descent given X, y, theta and no of iterations and learning rate alpha
#function returns an array of costs obtained in all iterations, and the final values of theta after grad descent
def gradientDescent(X,y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        h = X @ theta.T
        temp = h-y
        S = X.T @ temp
        S = (alpha/len(X))*S
        theta = theta - S.T
        c = computeCost(X,y,theta)
        cost[i] = c
        
    return cost,theta

#function to split input data and return training X training Y test X and test Y
def getTrainingTest(my_data):
    X = my_data.iloc[:,0:len(my_data.columns)-1]

    ones = np.ones([X.shape[0],1])
    X = np.concatenate((ones,X),axis=1)
    X = np.array(X, dtype=np.object_)
    
    y = my_data.iloc[:,len(my_data.columns)-1:len(my_data.columns)]
    y = np.array(y, dtype=np.object_)
    
    itrain = round(len(X)*0.8)
    
    trainX = X[0:itrain,:]
    testX = X[itrain+1:,:]
    
    trainY = y[0:itrain,:]
    testY = y[itrain+1:,:]

    return trainX, trainY, testX, testY

#To do: function to calculate root mean square error for accuracy of model
def getError(py,y):
    d = py-y
    s = d.T @ d 
    error = np.sqrt(s[0,0])
    return error

def costreg(X,y,theta,lamda):
    cost = computeCost(X,y,theta)
    th = theta[:,1:theta.shape[1]]
    th = lamda*th*th
    ones = np.ones(th.shape[1])
    term = th @ ones.T
    cost = cost + term

    return cost

#To do: function to do Gradient Descent with Regularization
def gdRegularized(X,y,theta,iters,alpha,lamda):
    cost = np.zeros(iters)
    term = 1 - lamda*(alpha/len(X))
    o = np.ones(theta.shape[1]-1)*term
    a = np.ones(1)
    M = np.concatenate((a,o),axis=0)
    
    for i in range(iters):
        h = X @ theta.T
        temp = h-y
        S = X.T @ temp
        S = (alpha/len(X))*S
        theta = theta * M
        theta = theta - S.T
        c = costreg(X,y,theta,lamda)
        cost[i] = c
    
    return cost,theta

def predict(X,theta):
    py = X @ theta.T
    return py

if len(sys.argv) < 2:
    print("usage python3 mvlr (filename)")
    sys.exit()
f = open('output.txt','w')
sys.stdout = f

filename = sys.argv[1]
my_data = pd.read_csv(filename)

my_data = (my_data - my_data.mean())/my_data.std()  #Normalizing the data
my_data.head()

trainX, trainY, testX, testY = getTrainingTest(my_data)
testY
theta = np.zeros([1,trainX.shape[1]],dtype=np.object_)

iters = 1000

ALPHA = [0.0001,0.001,0.01,0.1]
for alpha in ALPHA:
    theta = np.zeros([1,trainX.shape[1]],dtype=np.object_)
    cost,thita = gradientDescent(trainX,trainY,theta,iters,alpha)
    I=[i for i in range(iters)]
    plt.plot(I,cost,label=alpha)
    
    prY = predict(testX,thita)

    print("For alpha = ",alpha)
    print("Theta : ",thita)
    print("RMSE = ",getError(prY,testY))

print("\n")

lamdas = [0,1,10,100]

for lamda in lamdas:
    c,g = gdRegularized(trainX,trainY,theta,iters,alpha,lamda)

    prY1 = predict(testX,g)

    print("After regularization with lamda = ",lamda)
    print("Theta : ",g)
    print("Error = ",getError(prY1,testY))

plt.xlabel('Iterations')  
plt.ylabel('Cost') 
plt.legend()
plt.savefig('Cost_vs_iters.png')
