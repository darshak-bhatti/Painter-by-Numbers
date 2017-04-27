import os
import sys
import csv
import pandas as pd


files = os.listdir(sys.argv[1])
info = pd.read_csv('train_info.csv')
artists = info.artist
paintings = info.filename

D = {}
index = 0
for index in range(0,len(artists)):
  D[artists[index]] = 0

for index in range(0,len(artists)):
  D[artists[index]] += 1

i = 1
for key in D:
  print i, ", ", D[key]
  i += 1

