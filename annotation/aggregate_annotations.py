import csv
import pandas as pd
import sys
import matplotlib.pyplot as plt
import os

def aggregate_annotations(file):

    data = pd.read_csv(file)

    #Create Dataframe
    aggregation = {'Sample Name':[],
                    'Winner_annotation':[],
                    'Confidence':[],
                    'Number_annotations':[]
                    }
    
    df = pd.DataFrame(aggregation,columns = ['Sample_Name','Winner_annotation','Confidence','Number_annotations'])

    #Number_annotations
    num_anot = (pd.crosstab(data.Sample_name,data.Class))
    num_anot['sum'] = num_anot.sum(axis=1)
    num_anot = num_anot.reset_index()

    df['Number_annotations'] = num_anot['sum']
       
    #Confidence
    conf = (pd.crosstab(data.Sample_name,data.Class))
    res = conf.div(conf.sum(axis=1), axis=0)*100
    res = res.reset_index()
    
    #Values to Dataframe
    df['Sample_Name'] = res['Sample_name']
    sav=res['Sample_name']
    res=res.drop(['Sample_name'],axis=1)
    res['Max'] = res.idxmax(axis=1)
    res['max_value'] = res.max(axis=1)

    df['Winner_annotation'] = res["Max"]

    df['Confidence'] = res["max_value"]

    return df


def save_to_csv(df):   

    df.to_csv('aggregate_annotations.csv', index=False)

def report_annotations(file):

    data = pd.read_csv(file)
    df = aggregate_annotations(file)
    save_to_csv(df)

    #Total annotations
    print("\nTotal annotations:",df['Number_annotations'].sum())

    os.mkdir('plots')
    #Number of annotation that every user did + plot
    print("\nAnnotations of every user:\n",data['Username'].value_counts())
    user = data['Username'].value_counts()
    plot = user.plot(kind='pie', subplots=True, shadow = True,startangle=90,figsize=(15,10), autopct='%1.1f%%')
    plt.savefig("plots/pie.png")  
    plt.close()

    #Class distribution + plot
    print("Class Distribution\n",data['Class'].value_counts())
    conf = data['Class'].value_counts()
    conf.plot.bar()
    plt.xlabel('Classes')
    plt.ylabel('Number')
    plt.title('Class distribution')
    plt.tight_layout()
    plt.savefig('plots/classs_distr.png')

    #Number of annotations per file (save to csv)
    #per_movie = df[['Sample_Name','Number_annotations']]
    #per_movie.to_csv("anot_per_movie.csv",index=False)


report_annotations('annotations_database.txt')