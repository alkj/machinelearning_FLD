#!/usr/bin/env python
# coding: utf-8

# ## HW2: Linear Discriminant Analysis
# In hw2, you need to implement Fisher’s linear discriminant by using only numpy, then train your implemented model by the provided dataset and test the performance with testing data
# 
# Please note that only **NUMPY** can be used to implement your model, you will get no points by simply calling sklearn.discriminant_analysis.LinearDiscriminantAnalysis 

# ## Load data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


x_train = pd.read_csv("x_train.csv").values
y_train = pd.read_csv("y_train.csv").values[:,0]
x_test = pd.read_csv("x_test.csv").values
y_test = pd.read_csv("y_test.csv").values[:,0]

#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)


# ## 1. Compute the mean vectors mi, (i=1,2) of each 2 classes
print("task 1")
## Your code HERE
m1 = np.mean(x_train[y_train==0], axis=0 )
m2 = np.mean(x_train[y_train==1], axis=0 ) 
assert m1.shape == (2,)
print(f"mean vector of class 1: {m1}", f"mean vector of class 2: {m2}")
print("\n\n")



# ## 2. Compute the Within-class scatter matrix SW
print("task 2")
## Your code HERE
cl1 = np.zeros((2,2))
m1r = np.reshape(m1,(2,1))
for row in x_train[y_train==0]:
    r = np.reshape(row, (2,1))
    cl1 = cl1 + (r-m1r).dot((r-m1r).T)

cl2 = np.zeros((2,2))    
m2r = np.reshape(m2,(2,1))
for row in x_train[y_train==1]:
    r = np.reshape(row, (2,1))
    cl2 = cl2 + (r-m2r).dot((r-m2r).T)

sw = cl1+cl2
assert sw.shape == (2,2)
print(f"Within-class scatter matrix SW: \n{sw}")
print("\n\n")



# ## 3.  Compute the Between-class scatter matrix SB
print("task 3")
## Your code HERE
m = np.reshape(np.mean(x_train, axis=0 ), (2,1))
sb1 = (len(x_train))*(m1r-m).dot((m1r-m).T)
sb2 = (len(x_train))*(m2r-m).dot((m2r-m).T)
sb = sb1+sb2

assert sb.shape == (2,2)
print(f"Between-class scatter matrix SB: \n{sb}")
print("\n\n")



# ## 4. Compute the Fisher’s linear discriminant
print("task 4")
## Your code HERE
eigen_values, eigen_vectors = np.linalg.eig(np.linalg.pinv(sw).dot(sb))

pairs = []
pairs.append((np.abs(eigen_values[0]), eigen_vectors[:,0]))
pairs.append((np.abs(eigen_values[1]), eigen_vectors[:,1]))
pairs = sorted(pairs, key=lambda x: x[0], reverse=True) 
w = np.reshape(np.asarray(pairs[0][1]),(2,1))
assert w.shape == (2,1)
print(f" Fisher’s linear discriminant: {w}")
print("\n\n")


# ## 5. Project the test data by linear discriminant to get the class prediction by nearest-neighbor rule and calculate the accuracy score 
# you can use accuracy_score function from sklearn.metric.accuracy_score
print("task 5")

x_train_p = np.dot(x_train,w)
x_test_p = np.dot(x_test,w)


def nearest_neighbour(training_data_x, training_data_y, test):
    best_index = -1
    smallest_distance = 9999
    for index in range(len(training_data_x)):
        if smallest_distance>abs(training_data_x[index]-test):
            smallest_distance=abs(training_data_x[index]-test)
#            print("smallest distance ", smallest_distance)
            best_index = index
#    print("best index = ", best_index)
    return training_data_y[best_index]

correct = 0
for i in range(len(x_test_p)):
    if nearest_neighbour(x_train_p, y_train, x_test_p[i]) == y_test[i]:
        correct += 1
acc = correct/len(x_test_p)


print(f"Accuracy of test-set {acc}")

# ## 6. Plot the 1) projection line 2) Decision boundary and colorize the data with each class
# ### the result should be look like this [image](https://i2.kknews.cc/SIG=fe79fb/26q1000on37o7874879n.jpg ) (Red line: projection line, Green line: Decision boundary)
""" divide data """
x_test_tr = np.column_stack((x_test[:,0]*y_test,x_test[:,1]*y_test))
x_test_fl = np.column_stack((x_test[:,0]*-1*(y_test-1),x_test[:,1]*-1*(y_test-1)))
x_test_tr_wo = np.column_stack((np.delete(x_test_tr[:,0], np.argwhere(x_test_tr[:,0] == 0.0)),np.delete(x_test_tr[:,1], np.argwhere(x_test_tr[:,1] == 0.0))))
x_test_fl_wo = np.column_stack((np.delete(x_test_fl[:,0], np.argwhere(x_test_fl[:,0] == 0.0)),np.delete(x_test_fl[:,1], np.argwhere(x_test_fl[:,1] == 0.0))))


x_test_tr_wo_in = np.dot(x_test_p[y_test==1], w.T) + m1
x_test_fl_wo_in = np.dot(x_test_p[y_test==0], w.T) + m1


""" plot data """
plt.scatter(x_test_tr_wo[:,0], x_test_tr_wo[:,1], s=20, edgecolors='black', c='blue')
plt.scatter(x_test_fl_wo[:,0], x_test_fl_wo[:,1], s=20, edgecolors='black', c='red')
    
""" plot inverse values """
plt.plot(x_test_fl_wo_in[:,0], x_test_fl_wo_in[:,1], '.' , alpha=0.8, c='red')
plt.plot(x_test_tr_wo_in[:,0], x_test_tr_wo_in[:,1], '.' ,alpha=0.8, c='blue')

""" dimensions """
plt.gca().set_ylim(0, 5)
plt.gca().set_xlim(0, 4)

plt.show()
