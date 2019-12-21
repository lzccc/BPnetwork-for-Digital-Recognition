# -*- coding:utf-8 -*-
'''
Created on 2017年12月24日

@author: lenovo
'''

import os
import numpy as np
import pickle
import matplotlib
import math
import random
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy


def write_reult(value, name):
    with open(name,"w+") as f:
        for i in range(len(value)):
            f.write(str(value[i]))
            f.write("\n")

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / e_x.sum()
def relu_diff(x):
    x1 = copy.deepcopy(x)
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x1[i][j] > 0:
                x1[i][j] = 1
            else:
                x1[i][j] = 0
    return x1



# Definition of functions and parameters
# for example
EPOCH = 100

# Read all data from .pkl
(train_images, train_labels, test_images, test_labels) = pickle.load(open('./mnist_data/data.pkl', 'rb'),encoding='latin1')
print(len(train_images))

### 1. Data preprocessing: normalize all pixels to [0,1) by dividing 256
train_images = train_images/256
test_images = test_images/256
#print(type(train_images[1]))
### 2. Weight initialization: Xavier
hidden_layer1 = 300
hidden_layer2 = 100
output_layer = 10
hidden_layer1_w = np.zeros((len(train_images[0]), hidden_layer1))
hidden_layer2_w = np.zeros((hidden_layer1, hidden_layer2))
output_layer_w = np.zeros((hidden_layer2, output_layer))
hidden_layer1_b = np.zeros((1, hidden_layer1))
hidden_layer2_b = np.zeros((1, hidden_layer2))
output_layer_b = np.zeros((1, output_layer))
control1 = (6/(hidden_layer1+len(train_images[0])))**0.5
control2 = (6/(hidden_layer1 + hidden_layer2))**0.5
control3 = (6/(hidden_layer2 + output_layer))**0.5
for i in range(len(hidden_layer1_w)):
    for k in range(len(hidden_layer1_w[i])):    
        p = random.uniform(-control1, control1)
        hidden_layer1_w[i][k] = p
for i in range(len(hidden_layer2_w)):
    for k in range(len(hidden_layer2_w[i])):    
        p = random.uniform(-control2, control2)
        hidden_layer2_w[i][k] = p
for i in range(len(output_layer_w)):
    for k in range(len(output_layer_w[i])):    
        p = random.uniform(-control3, control3)
        output_layer_w[i][k] = p
print(hidden_layer2_w[0])

### 3. training of neural network

loss = np.zeros((EPOCH))
accuracy = np.zeros((EPOCH))
#print(loss)

for epoch in range(0, 100):
    #np.random.shuffle(train_images)
    

    #print(train_labels[10])
    t = []
    for i in range(len(train_labels)):
        a = np.zeros((10))
        a[train_labels[i]] = 1
        t.append(a)
    #print(t[10])        
    batches = []
    ts = []

    '''
    for i in range(5000):
        t
    for i in range(100):
        batches.append(train_images[i*100 : (i+1)*100])
        ts.append(t[i*100 : (i+1)*100])
    '''
    #print(epoch)
    for times in range(0,100):
        #print(times)
        k = random.randint(0,98)
        train_input = train_images[k*100 : (k+1)*100]
        ts = t[k*100 : (k+1)*100]
        #print(hidden_layer1_w[0])
        z1 = np.dot(train_input, hidden_layer1_w)+hidden_layer1_b
        z1 = np.maximum(z1, 0)
        #print(len(z1[0])) #100*300
        
        z2 = np.dot(z1, hidden_layer2_w)+hidden_layer2_b
        z2 = np.maximum(z2, 0) #100*100
        #print(len(z2))
        
        y = np.dot(z2, output_layer_w)+output_layer_b
        #print(y[0])
        for i in range(len(y)):
            y[i] = softmax(y[i])
        print(y[1])

        #print(y[1])
        losss = 0
        for i in range(100):
            for j in range(10):
                losss = losss - ts[i][j]*math.log(y[i][j])
        losss = losss/100
        #print(loss)
        
        theta3 = y - ts 
        #print(len(theta3[0])) #100*10
        rate = 0.1
        if epoch > 50:
            rate = 0.01
        output_layer_w_new = output_layer_w - rate*(1/100)*np.dot(z2.T, theta3) - rate*0.0005*output_layer_w
        output_layer_w_old = output_layer_w #100*10
        output_layer_w = output_layer_w_new
        hidden_layer1_w_new = []
        #print(theta3[0])
        #print(output_layer_w[0])
        sum_out_b = np.zeros((10))
        for i in range(100):
            #print(output_layer_b[i])
            #print(sum_out_b+output_layer_b[i])
            sum_out_b = sum_out_b + theta3[i]
        #print(sum_out_b)
        #print(output_layer_b[0])
        output_layer_b = output_layer_b - rate*(1/100)*sum_out_b
        fdiffz2 = relu_diff(z2)
        theta2 = np.dot(theta3, output_layer_w_old.T)*fdiffz2 #100*100
        output_layer_w_old = []
        hidden_layer2_w_new = hidden_layer2_w - rate*(1/100)*np.dot(z1.T, theta2) - rate*0.0005*hidden_layer2_w
        hidden_layer2_w_old = hidden_layer2_w #100*10
        hidden_layer2_w = hidden_layer2_w_new
        hidden_layer1_w_new = []
        sum_hidden2_b = np.zeros((100))
        for i in range(100):
            #print(output_layer_b[i])
            #print(sum_out_b+output_layer_b[i])
            sum_hidden2_b = sum_hidden2_b + theta2[i]
        hidden_layer2_b = hidden_layer2_b - rate*(1/100)*sum_hidden2_b
        #print(z1[50])
        fdiffz1 = relu_diff(z1)
        #print("aa")
        #print(fdiffz1[50])
        theta1 = np.dot(theta2, hidden_layer2_w_old.T)*fdiffz1
        hidden_layer1_w_old = []
        hidden_layer1_w_new = hidden_layer1_w - rate*(1/100)*np.dot(train_input.T, theta1) - rate*0.0005*hidden_layer1_w
        hidden_layer1_w_old = hidden_layer1_w #100*10
        hidden_layer1_w = hidden_layer1_w_new
        hidden_layer1_w_new = []
        hidden_layer1_w_old = []
        
        sum_hidden1_b = np.zeros((300))
        for i in range(100):
            #print(output_layer_b[i])
            #print(sum_out_b+output_layer_b[i])
            sum_hidden1_b = sum_hidden1_b + theta1[i]
        hidden_layer1_b = hidden_layer1_b - rate*(1/100)*sum_hidden1_b
        
        if times is 99:
            counter = 0
            loss[epoch] = losss
            print(losss)
            t_test = []
            for i in range(len(test_labels)):
                a = np.zeros((10))
                a[test_labels[i]] = 1
                t_test.append(a)
            #print(t[10])        
                
            for i in range(len(t_test)):    
                z1_test = np.dot(test_images[i], hidden_layer1_w)+hidden_layer1_b
                z1_test = np.maximum(z1_test, 0)
                #print(len(z1[0])) #100*300
                
                z2_test = np.dot(z1_test, hidden_layer2_w)+hidden_layer2_b
                z2_test = np.maximum(z2_test, 0) #100*100
                #print(len(z2))
                
                y_test = np.dot(z2_test, output_layer_w)+output_layer_b
            #print(y[0])
                for j in range(len(y_test)):
                    y_test[j] = softmax(y_test[j])    

                location = np.where(y_test == np.max(y_test))
         
                if int(location[1][0]) is int(test_labels[i]):
                    counter = counter+1
            print(counter)
            accuracy[epoch] = counter/1000
                
      
            

    # Forward propagation
    


    # Back propagation

    # Gradient update


    # Testing for accuracy


### 4. Plot
# for example
'''
plt.figure(figsize=(12,10))

plt.subplot(221)
#plt.xlabel('epoches')
plt.title('Train Loss')
plt.plot(loss)
plt.grid()
plt.tight_layout()
plt.subplot(224)
plt.title('TestAccuracy')
plt.xlabel('epoches')
plt.plot(accuracy)
plt.grid()
plt.tight_layout()
plt.savefig('figure.pdf', dbi=300)
'''

plt.figure(figsize=(12,10))

plt.subplot(211)
#plt.xlabel('epoches')
plt.title('Train Loss')
plt.plot(loss)
plt.grid()
plt.tight_layout()

plt.subplot(212)
plt.title('TestAccuracy')
plt.xlabel('epoches')
plt.plot(accuracy)
plt.grid()
plt.tight_layout()

plt.savefig('figure.pdf', dbi=500)
plt.show()


write_reult(loss,"loss.txt")
write_reult(accuracy,"accuracy.txt")