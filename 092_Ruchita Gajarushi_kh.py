#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
ser = pd.Series(['how', 'to', 'kick', 'ass?'])
ser.str.title()


# In[3]:


import numpy as np
import pandas as pd
ser = pd.Series(np.random.randint(1, 10, 7))

print(ser)


# In[6]:


import pandas as pd
ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04','2014-05-05', '2015-06-06T12:20'])
print("Original Series:",ser)
print("\nSeries of date strings to a timeseries:")
print(pd.to_datetime(ser))


# In[9]:


import pandas as pd
from collections import Counter
ser = pd.Series(['Apple', 'Orange', 'Plan', 'Python', 'Money'])
print("Original Series:",ser)
print("\nFiltered words:")
result = mask = ser.map(lambda c: sum([Counter(c.lower()).get(i, 0) for i in list('aeiou')]) >= 2)
print(ser[result])


# In[10]:


ser1 = pd.Series([10, 9, 6, 5, 3, 1, 12, 8, 13])
ser2 = pd.Series([1, 3, 10, 13])

[np.where(i == ser1)[0].tolist()[0] for i in ser2]

[pd.Index(ser1).get_loc(i) for i in ser2]


# In[11]:


import numpy as np
import pandas as pd

emails = pd.Series(['buying books at amazom.com', 'rameses@egypt.com', 'matt@t.co', 'narendra@modi.com'])

import re
pattern ='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,4}'
mask = emails.map(lambda x: bool(re.match(pattern, x)))
emails[mask]

emails.str.findall(pattern, flags=re.IGNORECASE)

[x[0] for x in [re.findall(pattern, email) for email in emails] if len(x) > 0]


# In[12]:


fruit = pd.Series(np.random.choice(['apple', 'banana', 'carrot'], 10))
weights = pd.Series(np.linspace(1, 10, 10))

weights.groupby(fruit).mean()


# In[13]:


ser = pd.Series([2, 10, 3, 4, 9, 10, 2, 7, 3])

dd = np.diff(np.sign(np.diff(ser)))
peak_locs = np.where(dd == -2)[0] + 1
peak_locs


# In[14]:


my_str = 'dbc deb abed gade'

ser = pd.Series(list('dbc deb abed gade'))
freq = ser.value_counts()
print(freq)
least_freq = freq.dropna().index[-1]
"".join(ser.replace(' ', least_freq))


# In[15]:


ser = pd.Series(np.random.randint(1,10,10), pd.date_range('2000-01-01', periods=10, freq='W-SAT'))
ser


# In[17]:


df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', chunksize=50)
df2 = pd.concat([chunk.iloc[0] for chunk in df], axis=1)
df2 = df2.transpose()
print(df2)


# In[18]:


df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', 
                 converters={'medv': lambda x: 'High' if float(x) > 25 else 'Low'})
print()


# In[19]:


df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', usecols=['crim', 'medv'])
print(df.head())


# In[20]:


df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')


df.loc[df.Price == np.max(df.Price), ['Manufacturer', 'Model', 'Type']]


row, col = np.where(df.values == np.max(df.Price))


df.iat[row[0], col[0]]
df.iloc[row[0], col[0]]


df.at[row[0], 'Price']
df.get_value(row[0], 'Price')


# In[22]:


import numpy as np
import pandas as pd
ser = pd.Series(np.random.randint(1, 10, 7))

print(ser)
a=np.argwhere(ser % 3==0)
print(a)


# In[ ]:




