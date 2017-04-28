'''
WRITTEN BY ERICK DRAAYER
ID: ecdraaye

This code preprocesses the dates in the csv files so that they are all numerical and if missing, 
selects a random number based on other dated works from the artist
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
artists = info.artist
artist_dates = {artists[k]: [] for k in range(len(artists))}

i = 0
for x in dates:
  if isinstance(x, basestring):
    dates[i] = re.search(r'\d+', x).group()
    artist_dates[artists[i]].append(dates[i])
  else:
    dates[i] = 0
    artist_dates[artists[i]].append(dates[i])
  i = i + 1

print artist_dates
newdata = info[['artist']]
dates = info.date
newdata['sameArtist'] = dates
newdata.to_csv('newdates.csv', index=False)

