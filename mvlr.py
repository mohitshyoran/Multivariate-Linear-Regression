import numpy as np
import pandas as pd
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

#To do: function to return training X training Y test X and test Y
#Use the 'extracting' lines of code and make changes
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
def getAccuracy(X,y,theta):

    return error

#To do: function to do Gradient Descent with Regularization
def gdRegularized(X,y,theta,iters,alpha):

    return cost,theta


#To do: get filename as argument from command line and replace 'home.txt'
my_data = pd.read_csv('home.txt',names=["size","bedroom","price"])

my_data = (my_data - my_data.mean())/my_data.std()  #Normalizing the data
my_data.head()

#extracting the features in array X. T
#To do: change the '2' to accomodate as many features(columns) as the input data has.
#Code should work for any size of input (any input file). Presently only in home.txt only 2 features.
X = my_data.iloc[:,0:2]                             

ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)                 #setting X[:,0] as 1's
#print(X)
#print(len(X))

#extracting predicted outputs
#To do: change the '2:3' to the last column of data (Generalized code).
y = my_data.iloc[:,2:3]                             

theta = np.zeros([1,3])                             #initializing theta values as zeroes 

alpha = 0.01
iters = 1000
#print(y)



print(computeCost(X,y,theta))



g,thita = gradientDescent(X,y,theta,iters,alpha)

c = computeCost(X,y,thita)
print("Final cost ",c)

fig, ax = plt.subplots()  
ax.plot(np.arange(iters), g, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch') 