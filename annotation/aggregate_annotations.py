import csv
import pandas as pd
import sys

import matplotlib.pyplot as plt

def get_stats(file):

  data = pd.read_csv(file)
  print("Annotations of every movie clip:\n", data['Sample_name'].value_counts())
  print("\n")

  clips = data['Sample_name'].value_counts().rename_axis('Sample_name').reset_index(name='Counts')
  clips_dict = dict(clips['Counts'].value_counts())

  # Groups of annotations
  for key, value in clips_dict.items():
      print("Num of files annotated by exactly %d annotators are: %d" %(key, value)) 

  #Number of annotations
  print("\nTotal number of annotations: ",data.shape[0],"\n")

  #Number of annotation that every user did
  print("Annotations of every user:\n",data['Username'].value_counts())
  user = data['Username'].value_counts()
  plot = user.plot(kind='pie', subplots=True, shadow = True,startangle=90,figsize=(15,10), autopct='%1.1f%%')
  plt.savefig("pie.png")  
  plt.close()

  #get confidence
  conf = (pd.crosstab(data.Sample_name,data.Class))

  #res = conf.div(conf.sum(axis=1), axis=0)*100
  conf = conf.reset_index()
  conf.to_csv('detailed_confidence.csv', index=False)

  #Class distribution
  
  print("Class Distribution\n",data['Class'].value_counts())
  conf = data['Class'].value_counts()
  conf.plot.bar()
  plt.xlabel('Classes')
  plt.ylabel('Number')
  plt.title('Class distribution')
  plt.tight_layout()
  plt.savefig('classs_distr.png')

  #Num of files NOT annotated 

  with open('videofiles.txt') as f:
    vidfiles = f.read().splitlines()

  print("Num of files not annotated: ",len(list(set(vidfiles) - set(data['Sample_name']))))


get_stats('annotations_database.txt')
