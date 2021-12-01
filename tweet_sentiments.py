import pandas as pd
import numpy as np
import math

data = pd.read_csv('training.1600000.processed.noemoticon.csv',encoding = 'ISO-8859-1',header= None)
data = data.to_numpy()



X= data[:,5].reshape((data.shape[0],1))
y= data[:,0].reshape((data.shape[0],1))

y = y.astype('float64')
d = {}
d4= {}
d0= {}

#class distributuion of the data
for i,k in zip(X,y):
    l = i[0].split
    for j in l:
        if k==0:
            if (j in d):
                d[j]+=1
                if (j in d0):
                    d0[j]+=1
                else:
                    d0[j]=1
            
            else:
                d[j]= 1
                if (j in d0):
                    d0[j]+=1
                else:
                    d0[j]=1
                
            
        elif k==4:
            if (j in d):
                d[j]+=1
                if (j in d4):
                    d4[j]+=1
                else:
                    d4[j]=1
            
            else:
                d[j]= 1
                if (j in d4):
                    d4[j]+=1
                else:
                    d4[j]=1
        else:
            continue

#parameters ki bat krte hai ab!
def phiY(y,m,n):
    
    m4 = len(y[y[ : , 0]==m, : ])
    m0 = len(y[y[ : , 0]==n, : ])
    prob = m4/(m0+m4)
    
    return prob
#class priors:
phiy4 = phiY(y,4,0)
phiy0 = phiY(y,0,4)
print(phiy4)
print(phiy0)

def numWords(d):
    sum = 0
    for k in d:
        sum+= d[k]
    return sum

numWordsd4 = numWords(d4)
m = len(d)
    

def phiK(dictionary,d,q,m):
    phik={}
    for k in dictionary:
        p = dictionary[k]+1
        logprob = math.log10(p/(q+m))
        phik[k]= logprob
    return phik
        
phik4 = phiK(d4,d,numWordsd4,m)
print(len(phik4))
numWordsd0 = numWords(d0)
phik0 = phiK(d0,d,numWordsd0,m)



#predict krne vale functions:
def conditional4(wordlist,phiK4,phiy4,m):

    probcond4 = math.log10(phiy4)
    for word in wordlist:
        if word in (phiK4):
            probcond4 += phiK4[word]
        else:
            probcond4 += math.log10(1/m)
    return probcond4

def conditional0(wordlist,phiK0,phiy0,m):

    probcond0 = math.log10(phiy0)
    for word in wordlist:
        if word in (phiK0):
            probcond0 += phiK0[word]
        else:
            probcond0 += math.log10(1/m)    
    return probcond0
    
    
#loading testcases
testdata = pd.read_csv('testdata.manual.2009.06.14.csv',encoding = 'ISO-8859-1',header= None)
testdata = testdata.to_numpy()

testX = testdata[:,5].reshape((testdata.shape[0],1))
tasty = testdata[:,0].reshape((testdata.shape[0],1))

#deleting case for neutral cases from tasty and testX
for i in range(tasty):
    if tasty[i]==2:
        np.delete(tasty,i,axis = 0)
        np.delete(testX,i,axis = 0)

Outy = []

for i in testX:
    l = i[0].split()
    asd4 = conditional4(l,phik4,phiy4,m)
    asd0 = conditional4(l,phik0,phiy0,m)
    if asd4>asd0:
        Outy.append(4)
    else:
        Outy.append(0)
        
Outy = np.array(Outy)
Outy = Outy.reshape((Outy.shape[0],1))
print(Outy)
    
#accuracy
        
    
    

































































