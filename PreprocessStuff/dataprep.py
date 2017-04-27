'''
WRITTEN BY ERICK DRAAYER
ID: ecdraaye

This code allows us to gather some statstics about the paintings such as how many paintings 
belong to each artists
'''

import pandas as pd
import math
from io import StringIO
from numbers import Number
import re

def isNaN(num):
    return num != num

info = pd.read_csv('all_data_info.csv')
dates = info.date
i = 0
for x in dates:
  if isinstance(x, basestring):
    print x
    info.date[i] = re.search(r'\d+', x).group()
    print re.search(r'\d+', x).group()
  else:
    print 0
  i = i + 1
  




