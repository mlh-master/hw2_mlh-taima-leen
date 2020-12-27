#!/usr/bin/env python
# coding: utf-8

# In[2]:


# BM 336546 - HW2
# Part I: Data Exploration


# In[2]:


# Loading Data

import pandas as pd
import numpy as np
from pathlib import Path
import random
# get_ipython().run_line_magic('load_ext', 'autoreload')


T1D_dataset = pd.read_csv("HW2_data.csv") 
T1D_features = T1D_dataset[['Age','Gender','Increased Urination','Increased Thirst','Sudden Weight Loss','Weakness','Increased Hunger',
                            'Genital Thrush','Visual Blurring','Itching','Irritability','Delayed Healing','Partial Paresis',
                            'Muscle Stiffness','Hair Loss','Obesity','Family History']]
Diagnosis = T1D_dataset[['Diagnosis']]

random.seed(10)  # fill your seed number here
print('hello')


# In[5]:


# replacing nans with samples

i = T1D_features.columns.values

T1Dc_features = {}
for x in i:
    Q = T1D_features[x]
    null_ind = np.where(Q.isnull())[0]

    for ii in null_ind:
        Q.iloc[ii] = Q[np.random.choice(np.where(Q.notnull())[0])]
    T1Dc_features[x] = Q
T1Dc_features=pd.DataFrame(T1Dc_features)
print(T1Dc_features)

print('hello')
