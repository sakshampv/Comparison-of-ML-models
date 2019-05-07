import csv
import pandas as pd
import numpy as np
import warnings
import math
#########################################################################
train_test_ratio = 0.8
k = 5
random_shuffle = 1
show_labels  = 0
num_iter = 30

dat = 4


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
    

def calc_dist(ob1, ob2):
    return np.sum(np.square(np.subtract(ob1[:-1], ob2[:-1])))

if random_shuffle ==1:
     df = df.sample(frac=1)
     df.index = range(len(df.index))

train_end = train_test_ratio*(len(df.index))
train_end = int(train_end)
test_strt = train_end+1
train = df.loc[0:train_end]
test = df.loc[test_strt:]


def calc_acc(a,b):
    q = np.abs(np.subtract(a,b))
    nums = q[np.nonzero(q)].size
    return 100-100*(nums/(a.size))

predicted_labels = []
for index, row in test.iterrows():
    dist =[]
    for index2, row2 in train.iterrows():
        dist += [calc_dist(row.values[:-1],row2.values[:-1])]
    df2 = pd.DataFrame()
    df2['dists'] = dist
    df2['labels'] = train['status']
    df2=df2.sort_values(by='dists')
    df2 = df2.values
    df2 = df2[:k]
    bb = np.array(np.unique(df2[:,1], return_counts=True)).T
    df3 = pd.DataFrame.from_records(bb)
    df3 = df3.sort_values(by=1, ascending = False)
    predicted_labels += [df3.loc[0][0]]
    

corr_labels = test['status'].values

print(calc_acc(corr_labels, predicted_labels))

   
    

num_classes = 7 
       
#####################CONFUSION MATRIX
test.index = np.arange(len(test.index))
predicted_te = []

for index, row in test.iterrows():
    dist =[]
    for index2, row2 in train.iterrows():
        dist += [calc_dist(row.values[:-1],row2.values[:-1])]
    df2 = pd.DataFrame()
    df2['dists'] = dist
    df2['labels'] = train['status']
    df2=df2.sort_values(by='dists')
    aaa = df2
    df2 = df2.values
    df2 = df2[:k]
    bb = np.array(np.unique(df2[:,1], return_counts=True)).T

    df3 = pd.DataFrame.from_records(bb)
    df3 = df3.sort_values(by=1, ascending = False)
    predicted_te += [df3.loc[0][0]]



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


predicted_tr = []

for index, row in train.iterrows():
    dist =[]
    for index2, row2 in train.iterrows():
        dist += [calc_dist(row.values[:-1],row2.values[:-1])]
    df2 = pd.DataFrame()
    df2['dists'] = dist
    df2['labels'] = train['status']
    df2=df2.sort_values(by='dists')
    df2 = df2.values
    df2 = df2[:k]
    bb = np.array(np.unique(df2[:,1], return_counts=True)).T
    df3 = pd.DataFrame.from_records(bb)
    df3 = df3.sort_values(by=1, ascending = False)
    predicted_tr += [df3.loc[0][0]]


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
    
    
    
    




