
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from tqdm import tqdm
import os

np.random.seed = 0

# pd.set_option('display.height', 1000)
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# pd.set_option('max_colwidth', 1000)
# pd.set_option('display.expand_frame_repr', False)


# In[2]:


in_f = '../data/data.tsv'
out_f = '../data/data_p.tsv'
chunksize = 10000
df = pd.read_csv(in_f, sep='\t', header=None, index_col=False)
df


# In[4]:


len(df)


# In[7]:


split=0.98
rows = len(df)

rows_train = int(np.floor(rows*split))
print(rows_train)

df.iloc[:rows_train,].to_csv('../data/traindata.tsv', sep='\t', index=False)
df.iloc[rows_train:,].to_csv('../data/validationdata.tsv', sep='\t', index=False)


# In[8]:


pd.read_csv('../data/validationdata.tsv', sep='\t', nrows=5)


# In[5]:


in_f = '../data/traindata.tsv'
out_f = '../data/traindata_p.tsv'
chunksize = 10000

def preprocess(in_f, out_f, chunksize):
    if os.path.exists(out_f):
        os.remove(out_f)
    reader = pd.read_csv(in_f, sep='\t', chunksize=chunksize, index_col=False, header=None)
    print('chunking')
    for chunk in tqdm(reader):
#         chunk.columns = ['query_id', 'query', 'passage_text', 'label', 'passage_id']
#         result1 = chunk[chunk.label==1]
        result1 = chunk[chunk.iloc[:,3]==1]
#         result0 = chunk[chunk.label==0].sample(frac =.11)
        result0 = chunk[chunk.iloc[:,3]==0].sample(frac =1/9)
        result = pd.concat([result1, result0], axis=0).sample(frac=1)
        result.to_csv(out_f, sep='\t',  mode='a', index=False, header=False)

preprocess(in_f, out_f, chunksize)


# In[4]:


in_f = '../data/validationdata.tsv'
out_f = '../data/validationdata_p.tsv'
chunksize = 10000

preprocess(in_f, out_f, chunksize)


# In[ ]:


# rows = len(train)/2
# split=0.9
# rows_train = int(np.floor(rows*split))
# df = pd.read_csv(out_f, nrows=rows, sep='\t',index_col=False)
# df.iloc[:rows_train,].to_csv('../data/traindata_p.tsv', sep='\t', index=False)
# df.iloc[rows_train:,].to_csv('../data/validationdata_p.tsv', sep='\t', index=False)
