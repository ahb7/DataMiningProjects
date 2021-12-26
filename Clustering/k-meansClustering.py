#This program was run on Spyder with Python 3.5
#k-means clustring algorithm is implemented and applied on 
#Wolberg's Breast Cancer dataset

import numpy as np
import pandas as pd
import scipy as sp
import random

#Initialize all the Algorithm specific parameters
#Threshold tau
tau = 0.01
#Number of clusters
k = 2
maxIteration = 50

#Define eucledian distance function
def distance(vect1, vect2):
    total = 0;
    #First attribute is patient id which is huge, which 
    #has bad impact on the distance function, hence starting from 2
    for i in range(2, len(vect1)):
        total += ((vect1[i] - vect2[i]) ** 2);        
    result = sp.sqrt(total)
    return result

#Read data csv file into a panda dataframe
df = pd.read_csv("CancerData.csv")

#print(df.head())
#print(df.shape)

#split columnwise between features and class
X = df.ix[:,0:11]
#print (X.head(5))

#Set total number of elements and number of attributes
n = X.shape[0]
col = X.shape[1]

#Create k unique centroids randomly
c = np.zeros(shape=(k,col))
#cPrev tracks values of centroids of previous iteration 
cPrev = np.zeros(shape=(k,col))

#Pickup c0 randomly
i=0
indx = random.randint(0, (n-1))
c[i] = X.ix[indx]
i += 1

#Pickup other k-1 centroids randomly
while (i<k):
    indx = random.randint(0, (n-1))
    c[i] = X.ix[indx]
    #Test if the centroid is unique
    for j in range(i):
        if ((c[i] == c[j]).all()):
            continue
    i +=1

#Create a list of two dimensional ndarrays to hold the clusters
cluster_list = []
for s in range(k):
    cluster_list.append(np.zeros(shape=(1, col)))

#count holds number of elements in each clusters
count = np.zeros(shape=(k))
converged = False

#Main for loop
for j in range(maxIteration):
    #Set each count to zero
    for s in range(k):
        count[s]= 0

    #Assign a datum to the nearest centroid
    for i in range(n):
        datum = X.ix[i]
        #Create an array to store distances between an element and centroids
        dist = np.zeros(shape=(k))
        min = float('inf')
        for s in range(k):
            dist[s] = distance(datum, c[s])
            if (min > dist[s]):
                min = dist[s]
                minIndx = s       
        
        if (count[minIndx] == 0):
            cluster_list[minIndx] = datum 
        else:
            cluster_list[minIndx] = np.append(cluster_list[minIndx], datum)          
        count[minIndx] += 1  
        cluster_list[minIndx] = (cluster_list[minIndx]).reshape(count[minIndx], col)
    
    #Set the cntroids to mean values of thier corresponding clusters
    for s in range(k):
        cPrev[s] = c[s]
        c[s] = np.mean(cluster_list[s], axis=0)
    
    #Check termination criterion
    distt = 0
    for s in range(k):
        distt += distance(cPrev[s], c[s])
    
    if ((distt/k) < tau):
        converged = True
        break
    else:
        #Clear the clusters for next iterations
        for s in range(k):
            cluster_list[s] = []
    #End main for loop


if (converged):
    
    print("\nOUTPUT - Here are the Clusters:")
    for s in range(k):
        print("\nNumber of elements in this Cluster: %d" % cluster_list[s].shape[0]) 
        print(cluster_list[s])
        
    #Report error by comparing cluster classifaction with test data label
    #Calculate error for each clusters and add to total error
    totalErr=0
    #We categorize a class by majority vote technique  
    print("\n")      
    for s in range(k):
        bi=0
        mi=0
        err=0
        for i in range(0, int(count[s])):
            cls = cluster_list[s][i][10]
            if (cls == 2):
                bi += 1
            else:
                mi += 1
        
        if (bi > mi):
            print("Cluster c%d is benign" % s)
            err = (mi/(bi+mi))
        else: 
            print("Cluster c%d is malignant" % s)
            err = (bi/(bi+mi))
        
        totalErr += err

    totalErrPercent = totalErr * 100
    print("Total Error Percentage: %f" % totalErrPercent)
    
else:
    print("k-means clustering could not converge!")
