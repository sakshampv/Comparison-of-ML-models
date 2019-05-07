import csv
import pandas as pd
import numpy as np
import random
import math

#########################################################################
train_test_ratio = 0.7
random_shuffle = 0   
weight_scale = 0.005
learning_rate = 0.1
num_iter = 40
dat = 1

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
if dat == 1:
    df = pd.read_table('./Haberman/haberman.data', delimiter = ',',header= None, names=['age', 'year', 'num_nodes', 'status'])

if dat == 2:
   df = pd.read_table('./Hepatitis/hepatitis.data', delimiter = ',',header= None, names=['status',0,1, 2,3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
   df = df[[0,1, 2,3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,'status']]
   df = df.replace('?', float('nan'))
   df = df.astype('float64')
   for i in range(19):
     kk = df[i].mean()
     df[i] = df[i].fillna(0)

if dat == 3:
    df = pd.read_table('./Heart/heart.data', delimiter = ' ',header= None, names=[1, 2,3, 4,5,6,7,8,9,10,11,12,13,'status'])
 
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
        
if dat == 5:
    df = pd.read_table('./Breast_Cancer/wdbc.data', delimiter = ',',header= None, names=['id', 'status', 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
    df = df.drop(['id'], axis = 1)
    for index, row in df.iterrows():
        if df.loc[index]['status'] == 'M':
             df.set_value(index,'status', 0)
        if df.loc[index]['status'] == 'B':
             df.set_value(index,'status', 1)     
    df = df[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,'status']]              
    


if random_shuffle ==1:
     df = df.sample(frac=1)
     df.index = range(len(df.index))

for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        df[feature_name] =(df[feature_name] - min_value) / (max_value - min_value)
   

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
    
       
       


################## PERFORMANCE MEASURES
predicted_tr = np.arange(len(train.index))
correct_cnt=0
tp=0
tn=0
fp=0
fn=0

for index, row in train.iterrows():

    if predicted_tr[index]<=0.5:
        predicted_tr[index] =0
    else:
       predicted_tr[index] = 1

    if int(row['status']) == int(predicted_tr[index]):
        if int(row['status'])==0:
            tn += 1
        else:
            tp += 1
    else:
        if int(row['status'])==0:
            fp += 1
        else:
            fn += 1
            
            

precision_tr = np.divide(tp,(tp+fp))
if math.isnan(precision_tr):
    precision_tr=0
if tp + fn == 0:
    recall_tr = 0
else:
     recall_tr = tp/(tp+fn)
if precision_tr+recall_tr==0:
 f1_tr = 0
else:     
    f1_tr = 2*(precision_tr*recall_tr)/(precision_tr+recall_tr)



predicted_te = np.arange(len(test.index))
correct_cnt=0
tp=0
tn=0
fp=0
fn=0
for index, row in test.iterrows():

    if predicted_te[index]<=0.5:
        predicted_te[index] =0
    else:
       predicted_te[index] = 1

    if int(row['status']) == int(predicted_te[index]):
        if int(row['status'])==0:
            tn += 1
        else:
            tp += 1
    else:
        if int(row['status'])==0:
            fp += 1
        else:
            fn += 1
            

precision_te = np.divide(tp,(tp+fp))
if math.isnan(precision_te):
    precision_te=0

recall_te = np.divide(tp,(tp+fn))
if precision_te+recall_te == 0:
    f1_te = 0
else:
    f1_te = 2*(precision_te*recall_te)/(precision_te+recall_te)

print("Precision_Training:",precision_tr)
print("Recall_Training:",recall_tr)
print("F1_Training:",f1_tr)
print("Precision_Test:",precision_te)
print("Recall_Test:",recall_te)
print("F1_Test:",f1_te)

            















