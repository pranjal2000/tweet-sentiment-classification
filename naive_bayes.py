

import pandas as pd
import numpy as np


# In[20]:


data = pd.read_csv('training.1600000.processed.noemoticon.csv',encoding = 'ISO-8859-1',header= None)

t=data.to_numpy()

a=(t[:,0])
a=a.reshape(a.shape[0],1)
#print(a)


# In[21]:


b=t[:,5]
b=b.reshape(b.shape[0],1)
#print(b)

test_array=np.append(a,b,axis=1)
#print(test_array)


# In[22]:


s=[[8,8]]
s=np.array(s)
d=(test_array.shape[0])
for i in range(d):
    if test_array[i,0]!=2:
        g=np.array([test_array[i,:]])
        #print(test_array[i,:])
        #print(np.array([test_array[i,:]]))
        s=np.append(s,g,axis=0)
s=s[1:,:]
print(test_array.shape)
#
test_array=s


# In[23]:


data1 = pd.read_csv("training.1600000.processed.noemoticon.csv")
t1=data1.to_numpy()

a=(t1[:,0])
a=a.reshape(a.shape[0],1)

b=t1[:,5]
b=b.reshape(b.shape[0],1)

print(a.shape)


# In[24]:


training_array=np.append(a,b,axis=1)
#print(training_array)
  
# In[16]:


e=0
f=0
for i in training_array:
    if i[0]==4:
        e=e+1
    f=f+1
prob_y_4=e/f

print("probability of getting 4 in trainingcase by random guessing = ",prob_y_4)
prob_y_0=1-prob_y_4
print("probability of getting 0 in trainingcase by random guessing = ",prob_y_0)


# In[17]:




# In[18]:


import math



dictt={}
dict0={} 
dict4={}
def some0(arr):
    for i in arr:
        if i in dict0:
            dict0[i]+=1
        else:
            dict0[i]=1
    
def some4(arr):
    for i in arr:
        if i in dict4:
            dict4[i]+=1
        else:
            dict4[i]=1
    
def some(arr):
    for i in arr:
        if i in dictt:
            dictt[i]+=1
        else:
            dictt[i]=1
    


dictt={}
dict0={}
dict4={}
c=0
for i in training_array:
    c+=1
    f=i[1].split()
    if i[0]==0:
        some0(f)
        some(f)
    else:
        some4(f)
        some(f)
#print(dictt['loll'])
probs_0={}
probs_4={}
print(dictt['loll'])

for i in dictt:
    probs_0[i]=1
    probs_4[i]=1

#print(dictt['loll'])
v=len(dictt)
print(v)
no_of_words_0=0
no_of_words_4=0
for i in training_array:
    for j in i[1].split():
        
        if i[0]==0:
            probs_0[j]=probs_0[j] +1
            no_of_words_0+=1
        else:
            probs_4[j]+=1
            no_of_words_4+=1
n0=no_of_words_0+v
n4=no_of_words_4+v
for i in probs_0:
    probs_0[i]=probs_0[i]/n0
    probs_4[i]=probs_4[i]/n4


print(8888)

prob_0_line={}
prob_4_line={}

cons0=math.log2(prob_y_0)
cons4=math.log2(prob_y_4)

for i in range(test_array.shape[0]):
    sumn=0.0
    sumnn=0.0
    for j in (test_array[i,1]).split():
        if j in dictt:
            sumn+=math.log2(probs_0[j])
            sumnn+=math.log2(probs_4[j])

    prob_0_line[i]=cons0+sumn
    prob_4_line[i]=cons4+sumnn
    


#for i in training_array:
#    some0(i[1].split())


# In[98]:


print(len(dict0))
print(len(dict4))


# In[99]:
#print(dict0['loll'])
#print(dict4['loll'])
#print(dictt['loll'])
# In[100]:
anomally=0

zerorahazero=0
zerobanafour=0
fourrahafour=0
fourbanazero=0
for i in prob_0_line:
    if test_array[i,0]==0:
        if prob_4_line[i]>prob_0_line[i]:
            anomally+=1
            zerobanafour+=1
        else:
            zerorahazero+=1
    elif test_array[i,0]==4:
        if prob_0_line[i]>prob_4_line[i]:
            anomally+=1
            fourbanazero+=1
        else:
            fourrahafour+=1

l=len(test_array)
ans=(l-anomally)/l

print("test set accuracy = ",ans)




prob_0_linee={}
prob_4_linee={}

cons00=math.log2(prob_y_0)
cons44=math.log2(prob_y_4)

for i in range(training_array.shape[0]):
    sumn=0.0
    sumnn=0.0
    for j in (training_array[i,1]).split():
        if j in dictt:
            sumn+=math.log2(probs_0[j])
            sumnn+=math.log2(probs_4[j])
    prob_0_linee[i]=cons00+sumn
    prob_4_linee[i]=cons44+sumnn

anomallyy=0

for i in prob_0_linee:
    if training_array[i,0]==0:
        if prob_4_linee[i]>prob_0_linee[i]:
            anomallyy+=1
    elif training_array[i,0]==4:
        if prob_0_linee[i]>prob_4_linee[i]:
            anomallyy+=1

l=len(training_array)
ans=(l-anomallyy)/l

print("training set accuracy = ",ans)

prob_test_0=0
prob_test_4=0

no_0=0
no_4=0
for i in test_array:
    if i[0]==0:
        no_0+=1
    elif i[0]==4:
        no_4+=1
prob_test_0=no_0/(no_0+no_4)
print("probability of getting 0 in testcase by random guessing = ",prob_test_0)

print("probability of getting 4 in testcase by random guessing = ",1-prob_test_0)

print("accuracy with majority prediction class, i.e. y equals 4, is= ",1-prob_test_0)
print(" ")
print("  actual class |")
print("___0___|___4___|______")
print(" ",zerorahazero," |  ",fourbanazero," | 0  pred-")
print(" ",zerobanafour,"  | ",fourrahafour," | 4  icted")
print(" ")


#%%
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer 




def create_filtered_sentence(sentence):
    stop_words = set(stopwords.words('english')) 
    word_tokens = word_tokenize(sentence) 
    filtered_sentence = [] 
    for w in word_tokens: 
        aaa=""
        for h in range(len(w)):
            if w[h]=='!' or w[h]=='@' or w[h]=='#' or w[h]=='$' or w[h]=='%' or w[h]=='&' or w[h]==',' or w[h]=='.' :
                aaaaaaaaa=""
            else:
                aaa+=w[h]
        w=aaa
        if not w in stop_words: 
            filtered_sentence.append(w) 

    filtered_sentence=np.array(filtered_sentence)
    ansss=[]
    for i in filtered_sentence:
        if i != "":
            ansss.append(i)
    ansss=np.array(ansss)
    ps = PorterStemmer() 
    for i in range(ansss.shape[0]):
        ansss[i]=ps.stem(ansss[i])
    ansss=np.array(ansss)
    return ansss

print(create_filtered_sentence("Ok, first gfh54 for assesment of the #kindle2 ...it fucking rocks!!!"))
#%%
print("a")
x=0
for i in training_array:
    x+=1
    if x==10000000:
        print("g")
    i[1]=create_filtered_sentence(i[1])
    #print(i)
print("a")
for i in test_array:
    i[1]=create_filtered_sentence(i[1])
print("a")
print(x)

print(training_array)

#%%
dictt={}
dict0={} 
dict4={}
def some0(arr):
    for i in arr:
        if i in dict0:
            dict0[i]+=1
        else:
            dict0[i]=1
    
def some4(arr):
    for i in arr:
        if i in dict4:
            dict4[i]+=1
        else:
            dict4[i]=1
    
def some(arr):
    for i in arr:
        if i in dictt:
            dictt[i]+=1
        else:
            dictt[i]=1
    


dictt={}
dict0={}
dict4={}
c=0
for i in training_array:
    c+=1
    f=i[1]
    ##f=create_filtered_sentence(f)
    if i[0]==0:
        some0(f)
        some(f)
    else:
        some4(f)
        some(f)



#print(dictt[''])


probs_0={}
probs_4={}
#print(dictt['loll'])

for i in dictt:
    probs_0[i]=1
    probs_4[i]=1

#print(dictt['loll'])
v=len(dictt)
print(v)
no_of_words_0=0
no_of_words_4=0
for i in training_array:
    for j in i[1]:
        
        if i[0]==0:
            probs_0[j]=probs_0[j] +1
            no_of_words_0+=1
        else:
            probs_4[j]+=1
            no_of_words_4+=1
n0=no_of_words_0+v
n4=no_of_words_4+v
for i in probs_0:
    probs_0[i]=probs_0[i]/n0
    probs_4[i]=probs_4[i]/n4


#print(8888)

prob_0_line={}
prob_4_line={}

cons0=math.log2(prob_y_0)
cons4=math.log2(prob_y_4)

for i in range(test_array.shape[0]):
    sumn=0.0
    sumnn=0.0
    for j in (test_array[i,1]):
        if j in dictt:
            sumn+=math.log2(probs_0[j])
            sumnn+=math.log2(probs_4[j])

    prob_0_line[i]=cons0+sumn
    prob_4_line[i]=cons4+sumnn
    


#for i in training_array:
#    some0(i[1].split())


# In[98]:


#print(len(dict0))
#print(len(dict4))


# In[99]:
#print(dict0['loll'])
#print(dict4['loll'])
#print(dictt['loll'])
# In[100]:
anomally=0

zerorahazero=0
zerobanafour=0
fourrahafour=0
fourbanazero=0
for i in prob_0_line:
    if test_array[i,0]==0:
        if prob_4_line[i]>prob_0_line[i]:
            anomally+=1
            zerobanafour+=1
        else:
            zerorahazero+=1
    elif test_array[i,0]==4:
        if prob_0_line[i]>prob_4_line[i]:
            anomally+=1
            fourbanazero+=1
        else:
            fourrahafour+=1

l=len(test_array)
ans=(l-anomally)/l

print("test set accuracy = ",ans)




prob_0_linee={}
prob_4_linee={}

cons00=math.log2(prob_y_0)
cons44=math.log2(prob_y_4)

for i in range(training_array.shape[0]):
    sumn=0.0
    sumnn=0.0
    for j in (training_array[i,1]):
        if j in dictt:
            sumn+=math.log2(probs_0[j])
            sumnn+=math.log2(probs_4[j])
    prob_0_linee[i]=cons00+sumn
    prob_4_linee[i]=cons44+sumnn

anomallyy=0

for i in prob_0_linee:
    if training_array[i,0]==0:
        if prob_4_linee[i]>prob_0_linee[i]:
            anomallyy+=1
    elif training_array[i,0]==4:
        if prob_0_linee[i]>prob_4_linee[i]:
            anomallyy+=1

l=len(training_array)
ans=(l-anomallyy)/l

print("training set accuracy = ",ans)






