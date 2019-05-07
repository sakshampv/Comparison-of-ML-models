import pandas as pd
import numpy as np
import random
import warnings
import math
#########################################################################


def calc_dist(ob1, ob2):
    return np.sum(np.square(np.subtract(ob1[:-1], ob2[:-1])))



train_test_ratio = 0.7
k = 8
random_shuffle = 1
show_labels  = 0
num_iter = 10

dat = 4

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


num_classes = 7

if random_shuffle ==1:
     df = df.sample(frac=1)
     df.index = range(len(df.index))


train_end = train_test_ratio*(len(df.index))
train_end = int(train_end)
test_strt = train_end+1
train = df.loc[0:train_end]
test = df.loc[test_strt:]
predicted_labels = []

choices = list(range(train_end+1))

random.shuffle(choices)
np.subtract(choices,1)
mask=choices[:k]
centres=train.loc[mask]
assigned= np.arange(train_end+1)
centres = centres.reindex(np.arange(k))

for j in range(num_iter):
  for index, row in train.iterrows():
    dist=10000
    for i in range(k):
        dist_=calc_dist(centres.loc[i].values,row)
        if dist_ < dist:
            assigned[index] =i
            dist = dist_
  nw_centres = centres
  for index, row in centres.iterrows():
     num = 0
     sum_vec = row
     sum_vec[:]=0
     for l in range(train_end+1):
        if assigned[l] == index:
            num += 1
            sum_vec = np.add(sum_vec,train.loc[assigned[l]])
     if num!=0:      
       sum_vec = np.divide(sum_vec, num)
       nw_centres.loc[index] = sum_vec
  centres = nw_centres  

test.index=np.arange(len(test.index))
predicted = np.arange(len(test.index))

for index, row in centres.iterrows():
    t = np.where(assigned==index)
    c=train.loc[t]['status'].values
    c = c.astype('int64')
    #unique_elements, counts_elements = np.unique(c, return_counts=True)
    #centres.loc[index]['status'] = np.max(np.asarray((unique_elements, counts_elements)))
    if len(c)==0:
        centres.loc[index]['status'] = centres.loc[index]['status']
    else:
        centres.loc[index]['status'] = np.bincount(c).argmax()
    
        
correct_cnt=0
for index, row in test.iterrows():

    dist = 10000
    for i in range(k):
         dist_ = calc_dist(row.values, centres.loc[i])
         if dist_ < dist:
               dist = dist_
               predicted[index] = centres.loc[i]['status']
    if int(predicted[index]) == int(row['status']):
        correct_cnt += 1

if show_labels == 1:
    print(predicted)
accuracy = 100*(correct_cnt/len(test.index))
print("Accuracy",accuracy)  


            
#####################CONFUSION MATRIX
predicted_te = np.arange(len(test.index))

for index, row in test.iterrows():

    dist = 10000
    for i in range(k):
         dist_ = calc_dist(row.values, centres.loc[i])
         if dist_ < dist:
               dist = dist_
               predicted_te[index] = centres.loc[i]['status']



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

    dist = 10000
    for i in range(k):
         dist_ = calc_dist(row.values, centres.loc[i])
         if dist_ < dist:
               dist = dist_
               predicted_tr[index] = centres.loc[i]['status']



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
    
    
    
    

    