#!/usr/bin/env python
# coding: utf-8

# In[602]:


import pandas as pd
import numpy as np
from random import randrange 


# In[603]:


column_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
df = pd.read_csv(r"data.csv", names=column_names)
df.head()


# In[604]:


# Change 'y' and 'n' to 1 and 0. And ? to NaN
df.replace(to_replace='y', value=1, inplace = True)
df.replace(to_replace='n', value=0, inplace = True)
df.replace(to_replace='?', value=np.nan, inplace = True)


# In[605]:


df.head()


# In[606]:


# Replace Nan with the mode
df.fillna(df.mode().iloc[0], inplace=True)


# In[607]:


df.head()


# In[608]:


# Change 1 and 0 to 'y' and 'n'
df.replace(to_replace=1, value='y', inplace = True)
df.replace(to_replace=0, value='n', inplace = True)


# In[609]:


df.head()


# In[610]:


# Shuffling the dataset
def shuffle(df):
    df = df.sample(frac = 1)
    list1 = list(range(435))
    df.reindex(range(435))
    return df
#df = shuffle(df)


# In[611]:


#splitting the Dataset into train and test
def split(df):
    type(df)
    l = list(df.count())
    train_n = int((80 / 100) * l[0])
    test_n = l[0] - train_n
    train = []
    test = []
    for i in range (train_n):
        row = []
        for j in range(17):
            index = randrange(train_n) 
            row.append(df.at[index, j])
        train.append(row)
    for i in range (test_n):
        row = []
        for j in range(17):
            index = randrange(test_n)
            row.append(df.at[index, j])
        test.append(row)
        
    n_demo = 0
    n_rep = 0
    for i in range(train_n):
        if train[i][16] == "democrat":
            n_demo += 1
        elif train[i][16] == "republican":
            n_rep += 1
    #print(n_demo)
    #print(n_rep)
    prob_demo = n_demo / train_n
    #print(prob_demo)
    prob_rep = n_rep / train_n
    #print(prob_rep)
    
    
    return train, test, prob_demo, prob_rep, n_demo, n_rep
d = split(df)
#print(d[0])


# In[612]:


# Finding probability of each column w.r.t democrat and republican
#train, test, prob_demo, prob_rep, n_demo, n_rep = split(df)
def prob(train):
    ci = []
    di = []
    ei = []
    fi = []
    for k in range(16):    
        c = 0
        d = 0
        e = 0
        f = 0
        for i in range(348):
            if train[i][16] == "democrat":
                if train[i][k] == "n":
                    c += 1   
                elif train[i][k] == "y":
                    e += 1
            elif train[i][16] == "republican":
                if train[i][k] == "n":
                    d += 1
                if train[i][k] == "y":
                    f += 1
        ci.append(c)
        di.append(d)
        ei.append(e)
        fi.append(f)
    return ei, ci, fi, di 
#prob(train)


# In[613]:


#train, test, prob_demo, prob_rep, n_demo, n_rep = split(df)
# Naive Bayes function
def nb(train, test, prob_demo, prob_rep):
    dy, dn, ry, rn = prob(train)
    term1 = []
    term2 = []
    
    for i in range(87):
        list = test[i]
        abc =[]
        
        for i in range(16):
            if list[i] == 'y':
                abc.append(dy[i])
            elif list[i] == 'n':
                abc.append(dn[i]) 
                
        result1 = 1
        for i in range(16):
            result1 *= (abc[i] / n_demo) 

        t1 = prob_demo * result1
        term1.append(t1)

        xyz =[]
        for i in range(16):
            if list[i] == 'y':
                xyz.append(ry[i])
            elif list[i] == 'n':
                xyz.append(rn[i])
                
        result2 = 1
        for i in range(16):
            result2 *= (xyz[i] / n_rep) 

        t2 = prob_rep * result2
        term2.append(t2)
    #print(term1)
    #print(term2)
    predicted = []
    for i in range(87):
        if (term1[i] < term2[i]):
            predicted.append("republican")
        else:
            predicted.append("democrat")
    return predicted
#nb(train, test, prob_demo, prob_rep)


# In[614]:


# Finding confusion matrix
#train, test, prob_demo, prob_rep, n_demo, n_rep = split(df)
#predicted = nb(train, test, prob_demo, prob_rep)
def conf_matr(test, predicted):
    cmatrix = [[0, 0], [0, 0]]
    for i in range(87):
        if test[i][16] == 'democrat':
            if predicted[i] == test[i][16]:
                cmatrix[0][0] += 1
            else:
                cmatrix[0][1] += 1
        elif test[i][16] == 'republican':
            if predicted[i] == test[i][16]:
                cmatrix[1][1] += 1
            else:
                cmatrix[1][0] += 1
    return cmatrix
#confusion_matrix = conf_matr(test, predicted)
#print(confusion_matrix)


# In[615]:


# Findind accuracy, recall, precission, F measure
def metrics(confusion_matrix):
    Total = confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[0][1] + confusion_matrix[1][0]
    #print(Total)
    accuracy = float(confusion_matrix[0][0] + confusion_matrix[1][1]) / Total
    print()
    print("Accuracy is :", accuracy)
    recall = float(confusion_matrix[0][0]) / (confusion_matrix[0][0] + confusion_matrix[1][1])
    print("Recall is :", recall)
    precision = float(confusion_matrix[0][0]) / (confusion_matrix[0][0]  + confusion_matrix[1][0] )
    print("Precision is", precision)
    f_measure = (2 * recall * precision) / (recall + precision)
    print("F measure is", f_measure)
    print()
    return accuracy


# In[623]:


def main(df):
    t_accuracy = []
    for i in range(5):
        #df = shuffle(df)
        s = split(df)
        train = s[0]
        test = s[1]
        prob_demo = s[2]
        prob_rep = s[3]
        n_demo = s[4]
        n_rep = s[5]
        predicted = nb(train, test, prob_demo, prob_rep)
        confusion_matrix = conf_matr(test, predicted)
        #print(confusion_matrix)
        print("For Dataset", i, ":")
        accuracy = metrics(confusion_matrix)
        t_accuracy.append(accuracy)
        
    accuracy = 0
    for i in range(5):
        accuracy += t_accuracy[i]
    accuracy = accuracy / 5
    print('\n\nAccuracy of Naive Bayes algorithm using 5-fold cross validation is:', accuracy)
main(df)


# In[ ]:





# In[ ]:




