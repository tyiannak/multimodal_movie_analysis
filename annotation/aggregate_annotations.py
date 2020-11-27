import shutil
import os
import numpy
import pandas as pd
import matplotlib.pyplot as plt


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


def save_to_csv(df):   

    df.to_csv('aggregate_annotations.csv', index=False)


def report_annotations(file):

    data = pd.read_csv(file)
    df = aggregate_annotations(file)
    save_to_csv(df)

    with open('videofiles.txt') as f:
        vidfiles = f.read().splitlines()
    
    print("\nTotal files available: ",len(vidfiles))

    sample_num = set(data['Sample_name'])
    #Total files annotated
    print("\nNum of files annotated: ", len(list(set(vidfiles) & sample_num)))

    #Num of files NOT annotated 
    print("\nNum of files not annotated: ",
          len(list(set(vidfiles) - sample_num)))

    #Total annotations
    print("\nTotal annotations:",df['Number_annotations'].sum())

    #Create directory for plots, if dir exists delete it
    if os.path.exists('plots'):
        shutil.rmtree('plots')
    os.mkdir('plots')

    #Number of annotation that every user did + plot
    print("\nAnnotations per user:\n",data['Username'].value_counts())
    user = data['Username'].value_counts()
    plot = user.plot(kind='pie', subplots=True, shadow=True,
                     startangle=90,
                     figsize=(15,10),
                     autopct='%1.1f%%')
    plt.savefig("plots/pie.png")  
    plt.close()

    #Class distribution (before majority) + plot
    
    count = data['Class'].value_counts()
    count = count.to_frame()
    per = count.div(count.sum(axis=0))*100
    count['Percentage'] = per['Class']
    count['Percentage'] = pd.Series([round(val, 2)
                                     for val in count['Percentage']],
                                    index=count.index)
    count['Percentage'] = pd.Series(["{0:.2f}%".format(val)
                                     for val in count['Percentage']],
                                    index=count.index)

    print("\nInitial Class Distribution (before majority): "
          "\nTotal: %s \n%s" % (data['Class'].shape[0], count))

    conf = data['Class'].value_counts()
    conf.plot.bar()
    plt.xlabel('Classes')
    plt.ylabel('Number')
    plt.title('Class distribution')
    plt.tight_layout()
    plt.savefig('plots/classs_distr_before.png')

    #Class distribution (after majority) + plot
    count = df['Winner_annotation'].value_counts()
    count = count.to_frame()
    per = count.div(count.sum(axis=0))*100
    count['Percentage'] = per['Winner_annotation']
    count['Percentage'] = pd.Series([round(val, 2)
                                     for val in count['Percentage']],
                                     index=count.index)
    count['Percentage'] = pd.Series(["{0:.2f}%".format(val)
                                     for val in count['Percentage']],
                                    index=count.index)

    print("\nAggregated Class Distribution (after majority): "
          "\nTotal: %s \n%s"% (df['Winner_annotation'].shape[0], count))

    conf = df['Winner_annotation'].value_counts()
    conf.plot.bar()
    plt.xlabel('Classes')
    plt.ylabel('Number')
    plt.title('Class distribution')
    plt.tight_layout()
    plt.savefig('plots/classs_distr_after.png')

    # Average agreement (confidence): average of all
    # confidences with >=2 annotations

    ann_gr_1 = df[df['Number_annotations'] >= 1]
    count = ann_gr_1['Number_annotations'].count()
    print('\n1 annotations:%s %.2f%%' % (count,
                                         numpy.divide(count,
                                                      df['Number_annotations'].
                                                      sum())*100))

    ann_gr_2 = df[df['Number_annotations'] >= 2]
    count = ann_gr_2['Number_annotations'].count()
    print('2 annotations:%s %.2f%%' % (count,
                                       numpy.divide(count,
                                                    df['Number_annotations'].
                                                    sum())*100))

    ann_gr_3 = df[df['Number_annotations'] >= 3]
    count = ann_gr_3['Number_annotations'].count()
    print('3 annotations:%s %.2f%%'%(count,
                                     numpy.divide(count,
                                                  df['Number_annotations'].
                                                  sum())*100))

    ann_gr_4 = df[df['Number_annotations'] >= 4]
    count = ann_gr_4['Number_annotations'].count()
    print('4 annotations:%s %.2f%%'%(count,
                                     numpy.divide(count,
                                                  df['Number_annotations'].
                                                  sum())*100))

    ann_gr_2.to_csv('conf.csv', index=False)
    print("\nAverage agreement : %.2f%%" %ann_gr_2['Confidence'].mean())
    print("\n")
    

report_annotations('annotations_database.txt')