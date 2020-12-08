import os.path
from os import path
import numpy
import pandas as pd


def aggregate_annotations(file):

    data = pd.read_csv(file)

    # Create Dataframe
    aggregation = {'Sample Name': [],
                   'Winner_annotation': [],
                   'Confidence': [],
                   'Number_annotations': []}
    
    df = pd.DataFrame(aggregation,columns = ['Sample_Name',
                                             'Winner_annotation',
                                             'Confidence',
                                             'Number_annotations'])

    # Number_annotations
    num_anot = (pd.crosstab(data.Sample_name,data.Class))
    num_anot['sum'] = num_anot.sum(axis=1)
    num_anot = num_anot.reset_index()

    df['Number_annotations'] = num_anot['sum']
       
    # Confidence
    conf = (pd.crosstab(data.Sample_name,data.Class))
    res = conf.div(conf.sum(axis=1), axis=0)*100
    res = res.reset_index()
    
    # Values to Dataframe
    df['Sample_Name'] = res['Sample_name']
    sav=res['Sample_name']
    res=res.drop(['Sample_name'],axis=1)
    res['Max'] = res.idxmax(axis=1)
    res['max_value'] = res.max(axis=1)

    df['Winner_annotation'] = res["Max"]

    df['Confidence'] = res["max_value"]

    return df

def save_to_csv(df,name):   

    df.to_csv(name, index=False)

def find_statics(file):

    df = aggregate_annotations(file)

    ann_gr_2 = df[(df['Number_annotations'] >= 2)
                    & (df['Confidence'] == 100)
                    & (df['Winner_annotation'] == 'Static') ]

    save_to_csv(ann_gr_2,'find_statics.csv')

    return ann_gr_2

def find_non_statics(file):

    df = aggregate_annotations(file)

    ann_gr_2 = df[(df['Number_annotations'] >= 2)
                    & (df['Confidence'] == 100)
                    & (df['Winner_annotation'] != 'Static')
                     ]

    save_to_csv(ann_gr_2,'find_non_statics.csv')

    return ann_gr_2


def del_files(df,dir):

    names = df['Sample_Name']
    names = names.values.tolist()
    names = set(names) 
    
    owd = os.getcwd()
    os.chdir(dir)
    dele = 0

    for filename in os.listdir('.'):
        if filename not in names:
           try: 
               os.remove(filename)
               dele+=1
           except:
               print('File not found')

    print('Number of deleted files: ',dele)
    print('Number of files:', len([name for name in os.listdir('.') if os.path.isfile(name)]))        
    
    os.chdir(owd)

file = 'annotations_database.txt'
df = find_statics(file)
df2 = find_non_statics(file)
del_files(df,'static')
del_files(df2,'non_static')