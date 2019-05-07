import csv
import pandas as pd
import numpy as np
import random
import math

#########################################################################
train_test_ratio = 0.7
random_shuffle = 0   
weight_scale = 0.001
learning_rate = 0.1
num_iter = 10
dat = 4

# =============================================================================

#DATA SELECTION:#
# 1 Haberman
# 2 Hepatitis
# 3 Heart
# 4 Zoo
# 5 Breast_Cancer
# =============================================================================

###############################################################################
warnings.filterwarnings("ignore")

if dat == 4:
    df = pd.read_table('./Zoo/zoo.data', delimiter = ',', header = None)

    a = ['aardvark', 'antelope', 'bear', 'boar', 'buffalo', 'calf',
                  'cavy', 'cheetah', 'deer', 'dolphin', 'elephant',
                  'fruitbat', 'giraffe', 'girl', 'goat', 'gorilla', 'hamster',
                  'hare', 'leopard', 'lion', 'lynx', 'mink', 'mole', 'mongoose',
                  'opossum', 'oryx', 'platypus', 'polecat', 'pony',
                  'porpoise', 'puma', 'pussycat', 'raccoon', 'reindeer',
                  'seal', 'sealion', 'squirrel', 'vampire', 'vole', 'wallaby','wolf']
    b= ['chicken', 'crow', 'dove', 'duck', 'flamingo', 'gull', 'hawk', 'kiwi', 'lark', 'ostrich', 'parakeet', 'penguin', 'pheasant', 'rhea', 'skimmer', 'skua', 'sparrow', 'swan', 'vulture', 'wren']

    c = [ 'pitviper', 'seasnake', 'slowworm', 'tortoise', 'tuatara']
    d = ['bass', 'carp', 'catfish', 'chub', 'dogfish', 'haddock', 'herring', 'pike', 'piranha', 'seahorse', 'sole', 'stingray', 'tuna']
    e = ['frog', 'frog', 'newt', 'toad']
    f = [ 'flea', 'gnat', 'honeybee', 'housefly', 'ladybird', 'moth', 'termite', 'wasp']
    g = [ 'clam', 'crab', 'crayfish', 'lobster', 'octopus', 'scorpion', 'seawasp', 'slug', 'starfish', 'worm']


    for index, row in df.iterrows():
     if row[0] in a:
        df.set_value(index, 0, 0)
     if row[0] in b:
        df.set_value(index, 0, 1)
     if row[0] in c:
        df.set_value(index, 0, 2)
     if row[0] in d:
        df.set_value(index, 0, 3)
     if row[0] in e:
        df.set_value(index, 0, 4)
     if row[0] in f:
        df.set_value(index, 0, 5)
     if row[0] in g:
        df.set_value(index, 0, 6)     
        
        
        
    df = df[[ 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17,0  ]]  

    df.columns = [0, 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 ,'status']  
        
l1,l2,l3,l4,l5,l6 = [],[],[],[],[],[]
for index,row in df.iterrows():
    if row[12] == 0:
        l1 += [1]
        l2 += [0]
        l3 += [0]
        l4 += [0]
        l5 += [0]
        l6 += [0]        
    if row[12] == 2:
        l2 += [1]
        l3 += [0]
        l4 += [0]
        l1 += [0]
        l5 += [0]
        l6 += [0]        
    if row[12] == 4:
        l3 += [1] 
        l4 += [0]
        l1 += [0]
        l2 += [0]
        l5 += [0]
        l6 += [0]        
        
    if row[12] == 5:
        l4 += [1]
        l1 += [0]
        l2 += [0]
        l3 += [0]
        l5 += [0]
        l6 += [0]        
    if row[12] == 6:
        l4 += [0]
        l1 += [0]
        l2 += [0]
        l3 += [0]     
        l5 += [1]
        l6 += [0] 
    if row[12] == 8:
        l4 += [1]
        l1 += [0]
        l2 += [0]
        l3 += [0]
        l5 += [0]
        l6 += [1]    
        
df=df.drop([16],axis = 1)        
df[12] = l1
df[16] =l2
df[17]=l3
df[18] = l4
df[19]=l5
df[20] = l6
df = df[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,'status']]


if random_shuffle ==1:
     df = df.sample(frac=1)
     df.index = range(len(df.index))

train_end = train_test_ratio*(len(df.index))
train_end = int(train_end)
test_strt = train_end+1
train = df.loc[0:train_end]
test = df.loc[test_strt:]


def sigmoid(x):
    return 1/(1+np.exp(-x))

def h(x,w):
    return sigmoid(np.dot(x,w))


num_features = len(train.loc[0])


w = weight_scale*np.random.rand(num_features)
b = np.ones((len(train.index),num_features))
b[:,1:] = train.values[:,:-1]

for i in range(num_iter):
    
    for j in range(num_features):
        sm = 0
        for l in range(len(train.index)):
            sm += (h(b[l],w)-train.loc[l]['status'])*(b[l,j])
        w[j] = w[j] - learning_rate*(sm)
    
    predicted = np.arange(len(train.index),dtype = 'float64')
    corr = 0

    for index, row in train.iterrows():
      c = np.ones((num_features))
      c[1:] = row.values[:-1]
  
      predicted[index]=h(c,w)
      
    for index, row in train.iterrows():

      if predicted[index]<=0.5:
        predicted[index] =0
      else:
       predicted[index] = 1
      if row['status'] == predicted[index]:
       corr += 1
    train_acc = 100*(corr/len(train.index))

    print(train_acc)    





predicted = np.arange(len(test.index),dtype = 'float64')
corr = 0
test.index = np.arange(len(test.index))
for index, row in test.iterrows():
    c = np.ones((num_features))
    c[1:] = row.values[:-1]

    predicted[index]=h(c,w)



for index, row in test.iterrows():

    if predicted[index]<=0.5:
        predicted[index] =0
    else:
       predicted[index] = 1
    if row['status'] == predicted[index]:
       corr += 1
       

acc = 100*(corr/len(test.index))

print(acc)       
num_classes = 7 
       
#####################CONFUSION MATRIX
predicted_te = np.arange(len(test.index))

for index, row in test.iterrows():
    c = np.ones((num_features))
    c[1:] = row.values[:-1]

    predicted_te[index]=h(c,w)



for index, row in test.iterrows():

    if predicted_te[index]<=0.5:
        predicted_te[index] =0
    else:
       predicted_te[index] = 1  



actual_te = test['status']

cm = np.zeros((num_classes,num_classes))

for i in range(len(actual_te)):
    cm[actual_te[i],predicted_te[i]] += 1
    
recall_te = np.diag(cm) / np.sum(cm, axis = 1)
precision_te = np.diag(cm) / np.sum(cm, axis = 0)

recall_te[np.isnan(recall_te)] = 0
precision_te[np.isnan(precision_te)] = 0
recall_te = np.mean(recall_te)
precision_te = np.mean(precision_te)



predicted_tr = np.arange(len(train.index))

for index, row in train.iterrows():
    c = np.ones((num_features))
    c[1:] = row.values[:-1]

    predicted_tr[index]=h(c,w)



for index, row in train.iterrows():

    if predicted_tr[index]<=0.5:
        predicted_tr[index] =0
    else:
       predicted_tr[index] = 1  



actual_tr = train['status']

cm = np.zeros((num_classes,num_classes))

for i in range(len(actual_tr)):
    cm[actual_tr[i],predicted_tr[i]] += 1
    
recall_tr = np.diag(cm) / np.sum(cm, axis = 1)
precision_tr = np.diag(cm) / np.sum(cm, axis = 0)
recall_tr[np.isnan(recall_tr)] = 0
precision_tr[np.isnan(precision_tr)] = 0
recall_tr = np.mean(recall_te)
precision_tr = np.mean(precision_te)


if precision_te+recall_te == 0:
    f1_te = 0
else:
    f1_te = 2*(precision_te*recall_te)/(precision_te+recall_te)
    
if precision_tr+recall_tr == 0:
    f1_tr = 0
else:
    f1_tr = 2*(precision_tr*recall_tr)/(precision_tr+recall_tr)



print("Precision_Training:",precision_tr)
print("Recall_Training:",recall_tr)
print("F1_Training:",f1_tr)
print("Precision_Test:",precision_te)
print("Recall_Test:",recall_te)
print("F1_Test:",f1_te)
    
    
    
    






