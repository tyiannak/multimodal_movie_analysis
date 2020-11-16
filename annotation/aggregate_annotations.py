import csv
import pandas as pd
import sys

def get_stats(file):

    data = pd.read_csv(file)    

    #Number of annotations
    print("\nTotal number of annotations: ",data.shape[0],"\n")

    #Number of annotation that every user did
    print("Annotations of every user:\n",data[' Username'].value_counts())   
    
    #get confidence
    conf = (pd.crosstab(data.Sample_name,data.Class))
    res = conf.div(conf.sum(axis=1), axis=0)*100
    res = res.reset_index()
    res.to_csv('detailed_confidence.csv', index=False)

    #print results to txt file
    sav=res['Sample_name']
    res=res.drop(['Sample_name'],axis=1)
    res['Max'] = res.idxmax(axis=1)
    res['max_value'] = res.max(axis=1)
    res['Sample_name']=sav

    orig_stdout = sys.stdout
    f = open('confidence.txt', 'w')
    sys.stdout = f

    for index, row in res.iterrows():
        
        print('The video with name',row['Sample_name']," belongs ",row['max_value'], "% to class",row['Max'] )
        
    sys.stdout = orig_stdout
    f.close()
  
get_stats('annotations_database.txt')