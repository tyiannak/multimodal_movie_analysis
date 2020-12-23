"""
Usage example:

python3 train.py -v dataset/Aerial dataset/None -a SVM Decision_Trees

Available algorithms for traning: SVM, Decision_Trees, KNN

"""
import argparse
import os
import numpy as np
import sys
from numpy import mean, std
from scipy import stats
from pathlib import Path
from sklearn import model_selection, preprocessing
from sklearn.model_selection import cross_validate,cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from analyze_visual import dir_process_video
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score


def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Create Shot "
                                                 "Classification Dataset")
    parser.add_argument("-v", "--videos_path", required=True, action='append', nargs='+',
                        help="Videos folder path")
    parser.add_argument("-a", "--training_algorithms",required=True, action='append', nargs='+',
                        help="Training Algorithms")

    return parser.parse_args()

def feature_extraction(videos_path):
    """
    Extract features from videos and save them
     to numpy arrrays(using dir_process_video() function)
    :param videos_path: directory of videos
    :return: extracted features and names of extracted files
    """
    x={}
    name_of_files={}

    for folder in videos_path:
        
        if os.path.exists(os.path.join(folder,folder[8:]+'_features.npy')) and os.path.exists(os.path.join(folder,folder[8:]+'_video_files_list.npy')):
            
            x["x_{0}".format(folder[8:])] = np.load(folder[8:]+'_features.npy')
            name_of_files["paths_{0}".format(folder[8:])] = np.load(folder[8:]+'_video_files_list.npy')
    
        else:

            x["x_{0}".format(folder[8:])],name_of_files["paths_{0}".format(folder[8:])] =dir_process_video(folder, 2, True, True,True)
        
    return x, name_of_files

def data_preparation(x):
    """
    Prepare the data before the training process
    :param x: exracted features from videos
    :return: features and labels
    """
    x_all = np.empty((0,244),float)
    y=[]

    for key, value in x.items():
        
        x_all = np.append(x_all,value,axis=0)
        
        for i in range(value.shape[0]):
            y.append(str(key))
    
    print('Before standarization: \n',x_all)
    #Standarization
    scaler = StandardScaler()
    # fit and transform the data
    x_all = scaler.fit_transform(x_all)


    print('AFTER: \n',x_all)
    #Encode target labels with value between 0 and n_classes-1
    lb = preprocessing.LabelEncoder()
    y = lb.fit_transform(y)
    return x_all,y


def train_models(x,training_algorithms):
    """
    Train the given algorithms and print accuracy,precision,recall
    :param x: features
    :training_algorithms: algorithm/s for training
    """

    x_all, y = data_preparation(x)

    scoring = ['precision_macro','recall_macro','accuracy']
    models = []

    for algorithm in training_algorithms:
        
        if algorithm == 'SVM':

            models.append(('SVM', SVC()))
            
        elif algorithm == 'Decision_Trees':
            
            models.append(('CART', DecisionTreeClassifier()))
            
        else:
 
            models.append(('KNN', KNeighborsClassifier()))

    
    scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}

    results=[]
    names = []

    for name,model in models:

        #Slit data to train/test
        kfold = model_selection.KFold(n_splits=10, random_state=1, shuffle=True)
    
        scores = model_selection.cross_validate(model,
                                X = x_all,
                                y = y.ravel(),
                                scoring=scoring,
                                cv=kfold, n_jobs=-1)

        results.append(scores)
        names.append(name)
        print('---------') 
        for key,values in scores.items():
            msg = "%s: %s, %f (%f)" % (name, key, values.mean(), values.std())
            print(msg)
        print('---------')            

if __name__ == "__main__":

    args = parse_arguments()
    videos_path = args.videos_path
    training_algorithms = args.training_algorithms

    #Convert list of lists to a single list
    videos_path = [item for sublist in videos_path for item in sublist]
    training_algorithms = [item for sublist in training_algorithms for item in sublist]

    for paths in videos_path:
        assert os.path.exists(paths), "Video Path doesn't exist, " + \
                                    str(paths)

    available_algorithms = ['SVM', 'Decision_Trees', 'KNN']

    for algorithm in training_algorithms:
        if algorithm not in available_algorithms:

            sys.exit('%s is not available please read the Usage example'%(algorithm))

    #Extract features of videos
    x, name_of_files = feature_extraction(videos_path)

    #Train the models
    train_models(x,training_algorithms)

