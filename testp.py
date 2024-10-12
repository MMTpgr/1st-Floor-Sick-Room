import torch
import torch.nn as nn
from torch.nn import functional as F


block_size=8
batch_size=4 

with open('pg2.txt','r',encoding='utf-8') as f:
    t= f.read()
    cd = sorted(set(t))
    vocab_size = len(cd)

string_to_int = { ch:i for i,ch in enumerate(cd) }
int_to_string = { i:ch for i,ch in enumerate(cd) }


encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

data = torch.tensor(encode(t),dtype=torch.long)



n= 500
train_data = data[:n]
val_data = data[n:]

bs = 8 
x = train_data[:bs]
y = train_data[1:bs+1]

r = torch.randint(-199,100,(5,))
print(r)

def get_batch(split):
    if (split=='train'):
        data = train_data
    else:
        data  = val_data
    
    ix = torch.randint(len(data)-block_size,(batch_size,))
    print(ix)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x,y

x,y = get_batch('train')
print('inputs:',x)
print('target:',y)
    



























