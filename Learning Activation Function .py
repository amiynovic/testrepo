#!/usr/bin/env python
# coding: utf-8

# # Learning Activation Function 

# In[ ]:


pip install keras


# In[5]:


from keras.models import Sequential


# In[3]:


import keras


# In[7]:


from keras.layers import Dense, Activation, Dropout


# In[9]:


from keras.optimizers import SGD


# In[1090]:


#Import libraries

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[221]:


def weights_init(layers_dim):
    params = {}
    
    n = len(layers_dim)
    for i in range(1, n):
        params['W' + str(i)] = np.random.randn(layers_dim[i], layers_dim[i-1])*0.01
        params['b' + str(i)] = np.zeros((layers_dim[i], 1))
    return params


# #### définir la fonction d'activation Relu et sigmoid

# In[222]:


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


# In[223]:


int(len(params)/2)


# In[1192]:


def forward(X, params):
    # intermediate layer use relu as activation
    # last layer use sigmoid
    n_layers = int(len(params)/2)
    A = X
    cache = {}
    for i in range(1, n_layers):
        W, b = params['W'+str(i)], params['b'+str(i)]
        Z = np.dot(W, A) + b
        A = relu(Z)
        cache['Z'+str(i)] = Z
        cache['A'+str(i)] = A
    
    # last layer
    W, b = params['W'+str(i+1)], params['b'+str(i+1)]
    Z = np.dot(W, A) + b
    A = sigmoid(Z)
    cache['Z'+str(i+1)] = Z
    cache['A'+str(i+1)] = A
    
    return cache, A 


# In[1210]:





# #### Cost function

# In[1191]:


def compute_cost(A, Y):
    """
    For binary classification, both A and Y would have shape (1, m), where m is the batch size
    """
    assert A.shape == Y.shape
    m = A.shape[1]
    s = np.dot(Y, np.log(A.T)) + np.dot(1-Y, np.log((1 - A).T))
    loss = -s/m
    return np.squeeze(loss)


# In[1190]:


def sigmoid_grad(A, Z):
    grad = np.multiply(A, 1-A)
    return grad


def relu_grad(A, Z):
    grad = np.zeros(Z.shape)
    grad[Z>0] = 1
    return grad


# In[1189]:



def backward(params, cache, X, Y):
    """
    params: weight [W, b]
    cache: result [A, Z]
    Y: shape (1, m)
    """
    grad = {}
    n_layers = int(len(params)/2)
    m = Y.shape[1]
    cache['A0'] = X
    
    for l in range(n_layers, 0, -1):
        A, A_prev, Z = cache['A' + str(l)], cache['A' + str(l-1)], cache['Z' + str(l)]
        W = params['W'+str(l)]

        if l == n_layers:
            dA = -np.divide(Y, A) + np.divide(1 - Y, 1 - A)
        
        if l == n_layers:
            dZ = np.multiply(dA, sigmoid_grad(A, Z))
        else:
            dZ = np.multiply(dA, relu_grad(A, Z))
            
        dW = np.dot(dZ, np.transpose(A_prev))/m
        db = np.sum(dZ, axis=1)/m
        dA = np.dot(np.transpose(W), dZ)
        dq=

        grad['dW'+str(l)] = dW
        grad['db'+str(l)] = db
    
    return grad


# In[228]:


def optimize(params, grads, lr):
    n_layers = int(len(params)/2)
    for i in range(1, n_layers+1):
        dW, db = grads['dW'+str(i)], grads['db'+str(i)]
        params['W'+str(i)] -= lr*dW
        params['b'+str(i)] -= lr*db
    return params


# In[346]:


from sklearn import datasets


X, y = datasets.make_classification(n_samples=10000, n_features=200, random_state=123)

X_train, X_test = X[:8000], X[8000:]
y_train, y_test = y[:8000], y[8000:]

print('train shape', X_train.shape)
print('test shape', X_test.shape)


# In[328]:


print('train shape', y_train.shape)
print('test shape', y_test.shape)


# #### training section

# In[912]:


layers_dim=[15,3,1]


# In[913]:


params=weights_init(layers_dim)


# In[910]:


params


# In[914]:


cache, A=forward(np.transpose(X_train), params) 


# In[915]:


cache


# In[349]:


y_train=np.asmatrix(y_train)


# In[350]:


y_train.shape


# In[235]:


A.shape


# In[241]:


compute_cost(A, y_train)


# In[242]:


grads=backward(params, cache, np.transpose(X_train), y_train)


# In[243]:


grads['dW3'].shape


# In[244]:


np.sum([1,2,3])


# In[139]:


X_train.shape


# In[162]:


new_params=optimize(params, grads, 0.1)


# In[163]:


new_params


# In[164]:


cache, A=forward(np.transpose(X_train), params) 


# In[165]:


compute_cost(A, y_train)


# ### Algorithme Backpropagation

# In[1193]:


Loss=[]


# In[1194]:


P=weights_init([15,3,1])


# In[1195]:


len(P)


# In[1196]:


for i in range(1000):
    cache, A=forward(np.transpose(X_train),P)
    grads=backward(P,cache,np.transpose(X_train),y_train)
    P=optimize(P,grads,0.1)
    Loss.append(compute_cost(A, y_train))
               
   


# In[1302]:


Loss=np.asarray(Loss)


# In[1198]:


P


# In[843]:


Loss.shape


# #### Draw loss function

# In[1276]:


x=[]


# In[1277]:


for i in range(0,1000):
    x.append(i)


# In[1278]:


np.asarray(x).shape


# In[1279]:


y=[]


# In[1280]:


for i in range(0,1000):
    y.append(Loss[i,0])


# In[1281]:


np.asarray(y).shape


# In[850]:


plt.plot(x,y)


# In[723]:


P['W1'].shape


# In[724]:


cache_pred, y_pred=forward(np.transpose(X_test),P)


# In[725]:


y_pred.shape


# In[726]:


y_test.shape


# ### F1-Score  accuarcy

# In[ ]:


n=y_test.shape[1]
TP=0
FN=0
FP=0

for i in range(n):
    if (y_pred[0,i]>0.5) and (y_test[0,i]==1):
        TP=TP+1
for i in range(n):
    if (y_pred[0,i]<=0.5) and (y_test[0,i]==1):
        FN=FN+1
for i in range(n):
    if (y_pred[0,i]>0.5) and (y_test[0,i]==0):
        FP=FP+1
precision=TP/(TP+FP)
recall=TP/(TP+FN)
F1=(2*precision*recall)/(precision+recall)
F1


# In[ ]:





# In[ ]:





# In[273]:


np.transpose(X_train).shape[0]


# In[332]:


X_train.shape[0]


# In[335]:


np.transpose(y_train).shape[0]


# ## 2. Application on dataset

# In[643]:


dt=pd.read_csv("C:/Users/pc/Desktop/bank.csv",delimiter=";")


# In[644]:


dt.shape


# In[645]:


dt.dtypes


# In[646]:


dt['y'].replace(to_replace=['yes','no'], value=[1,0], inplace=True)


# In[647]:


dt.groupby(['contact'])['y'].value_counts(normalize=True)


# In[648]:


dt["job"]=dt["job"].astype('category')
dt["job"]=dt["job"].cat.codes


# In[649]:


dt["marital"]=dt["marital"].astype('category')
dt["marital"]=dt["marital"].cat.codes


# In[650]:


dt["education"]=dt["education"].astype('category')
dt["education"]=dt["education"].cat.codes


# In[651]:


dt["default"]=dt["default"].astype('category')
dt["default"]=dt["default"].cat.codes


# In[652]:


dt["housing"]=dt["housing"].astype('category')
dt["housing"]=dt["housing"].cat.codes


# In[653]:


dt["loan"]=dt["loan"].astype('category')
dt["loan"]=dt["loan"].cat.codes


# In[654]:


dt["contact"]=dt["contact"].astype('category')
dt["contact"]=dt["contact"].cat.codes


# In[656]:


dt["month"]=dt["month"].astype('category')
dt["month"]=dt["month"].cat.codes


# In[657]:


dt.dtypes


# In[658]:


X=dt[['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous']]


# In[659]:


y=dt['y']


# In[660]:


from  sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[661]:


X=StandardScaler().fit_transform(X)


# In[662]:


# Split data to train and test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)


# In[663]:


X_train.shape


# In[664]:


y_train=np.asmatrix(y_train)


# In[665]:


y_train.shape


# ## 3. Learning Activation Function

# In[1468]:


def weights_init2(layers_dim):
    params = {}
    
    n = len(layers_dim)
    params['q'] = np.random.randn(1, 3)*0.01
    for i in range(1, n):
        params['W' + str(i)] = np.random.randn(layers_dim[i], layers_dim[i-1])*0.01
        params['b' + str(i)] = np.zeros((layers_dim[i], 1))
        
    return params


# In[1447]:


A=[[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
B=[2,0,2]
np.multiply(np.transpose(B),A)


# In[1469]:


def sigmoid2(x):
    return 1/(1 + np.exp(-np.multiply(q,x)))


# In[1470]:


def forward2(X, params):
    # intermediate layer use relu as activation
    # last layer use sigmoid
    n_layers = int(len(params)/2)
    A = X
    cache = {}
    for i in range(1, n_layers):
        W, b = params['W'+str(i)], params['b'+str(i)]
        Z = np.dot(W, A) + b
        A = relu(Z)
        cache['Z'+str(i)] = Z
        cache['A'+str(i)] = A
    
    # last layer
    W, b = params['W'+str(i+1)], params['b'+str(i+1)]
    Z = np.dot(W, A) + b
    Z=np.multiply(np.transpose(params['q']),Z)
    A = sigmoid(Z)
    cache['Z'+str(i+1)] = Z
    cache['A'+str(i+1)] = A
    
    return cache, A 


# In[1471]:


def compute_cost2(A, Y):
    """
    For binary classification, both A and Y would have shape (1, m), where m is the batch size
    """
    assert A.shape == Y.shape
    m = A.shape[1]
    s = np.dot(Y, np.log(A.T)) + np.dot(1-Y, np.log((1 - A).T))
    loss = -s/m
    return np.squeeze(loss)


# In[1411]:


def sigmoid_grad2(A, Z, s):
    grad =s*np.multiply(A, 1-A)
    return grad


# In[1472]:


def backward2(params, cache, X, Y):
    """
    params: weight [W, b]
    cache: result [A, Z]
    Y: shape (1, m)
    """
    grad = {}
    n_layers = int(len(params)/2)
    m = Y.shape[1]
    cache['A0'] = X
    
    for l in range(n_layers, 0, -1):
        A, A_prev, Z = cache['A' + str(l)], cache['A' + str(l-1)], cache['Z' + str(l)]
        W = params['W'+str(l)]

        if l == n_layers:
            dA = -np.divide(Y, A) + np.divide(1 - Y, 1 - A)
        
        if l == n_layers:
            dZ = np.multiply(dA, sigmoid_grad(A, Z))
        else:
            dZ = np.multiply(dA, relu_grad(A, Z))
            
        dW = np.dot(dZ, np.transpose(A_prev))/m
        db = np.sum(dZ, axis=1)/m
        dA = np.dot(np.transpose(W), dZ)
       

        grad['dW'+str(l)] = dW
        grad['db'+str(l)] = db
        
    
    return grad


# In[1474]:


def optimize2(params, grads, lr):
    n_layers = int(len(params)/2)
    params['q'] = params['q'] +lr
    for i in range(1, n_layers+1):
        dW,db = grads['dW'+str(i)], grads['db'+str(i)]
        params['W'+str(i)] -= lr*dW
        params['b'+str(i)] -= lr*db
        
    return params


# ### Deep learning with Activation function learning

# In[1475]:


Loss=[]


# In[1476]:


P=weights_init2([15,3,1])


# In[1422]:


q=abs(np.random.randn())


# In[1462]:


P['q']


# In[1477]:


P['q'][0,0]


# In[1478]:


for i in range(600):
    cache, A=forward2(np.transpose(X_train),P)
    grads, dq=backward2(P,cache,np.transpose(X_train),y_train)
    P=optimize2(P,grads,0.1)
    Loss.append(compute_cost(A, y_train))


# In[1425]:


cache, A=forward2(np.transpose(X_train),P,q)


# In[1158]:


grad=backward2(P,cache,np.transpose(X_train),y_train)


# In[1164]:


grad


# In[1211]:


cache, A=forward2(np.transpose(X_train),P)


# In[1377]:


q


# In[1310]:


cache_pred, y_pred=forward2(np.transpose(X_test),P,q)


# In[1366]:


Loss=np.asarray(Loss)


# In[1367]:


x=[]
for i in range(0,600):
    x.append(i)


# In[1368]:


y=[]
for i in range(0,600):
    y.append(Loss[i,0])


# In[ ]:





# In[ ]:





# In[1369]:


plt.plot(x,y)


# In[1376]:


n=y_test.shape[1]
TP=0
FN=0
FP=0

for i in range(n):
    if (y_pred[0,i]>0.5) and (y_test[0,i]==1):
        TP=TP+1
for i in range(n):
    if (y_pred[0,i]<=0.5) and (y_test[0,i]==1):
        FN=FN+1
for i in range(n):
    if (y_pred[0,i]>0.5) and (y_test[0,i]==0):
        FP=FP+1
precision=TP/(TP+FP)
recall=TP/(TP+FN)
F1=(2*precision*recall)/(precision+recall)
F1


# In[ ]:




