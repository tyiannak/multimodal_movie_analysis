"""
Usage example:

python3 train.py -v dataset/Aerial dataset/None -a SVM Decision_Trees

Available algorithms for traning: SVM, Decision_Trees, KNN, Adaboost, Extratrees, RandomForest

"""
import warnings
import argparse
import os
import numpy as np
import sys
import fnmatch
import itertools
from sklearn import model_selection, preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, \
    RandomForestClassifier
from analyze_visual import dir_process_video
from sklearn.metrics import make_scorer, accuracy_score, precision_score, \
    recall_score, f1_score, plot_confusion_matrix, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split


def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Create Shot "
                                                 "Classification Dataset")
    parser.add_argument("-v", "--videos_path", required=True, action='append',
                        nargs='+', help="Videos folder path")
    parser.add_argument("-a", "--training_algorithms", required=True,
                        action='append', nargs='+', help="Training Algorithms")

    return parser.parse_args()


def feature_extraction(videos_path):
    """
    Extract features from videos and save them
     to numpy arrrays(using dir_process_video() function)
    :param videos_path: directory of videos
    :return: extracted features and names of extracted files
    """
    x = {}
    name_of_files = {}

    for folder in videos_path:
        # for each class-folder

        # get list of np files in that folder (where features can have
        # been saved):
        np_feature_files = fnmatch.filter(os.listdir(folder), '*_features.npy')
        print(np_feature_files)
        # if feature npy files exist:
        if len(np_feature_files) > 0:
            for file in np_feature_files:
                if os.path.isfile(os.path.join(folder, file)):
                    x["x_{0}".format(folder)] = np.load(os.path.join(folder,
                                                                     file))
        else:
            # calculate features for current folder:
            x["x_{0}".format(folder)],\
            name_of_files["paths_{0}".format(folder)] = \
                dir_process_video(folder, 2, True, True, True)   

    return x, name_of_files


def data_preparation(x):
    """
    Prepare the data before the training process
    :param x: exracted features from videos
    :return: features and labels
    """
    x_all = np.empty((0, 244), float)
    y = []

    for key, value in x.items():
        x_all = np.append(x_all,value,axis=0)
        for i in range(value.shape[0]):
            y.append(str(key))
    
    # Standarization
    scaler = StandardScaler()
    # fit and transform the data
    x_all = scaler.fit_transform(x_all)

    # Encode target labels with value between 0 and n_classes-1
    lb = preprocessing.LabelEncoder()
    y = lb.fit_transform(y)

    return x_all, y


def plot_confusion_matrix(name, cm, classes):
    """
    Plot confusion matrix
    :name: name of classifier
    :cm: estimates of confusion matrix
    :classes: all the classes
    """
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("Conf_Mat_"+str(name)+".jpg")


def Grid_Search_Process(classifier, grid_param, metrics, x_all, y):
    """
    Hyperparameter tuning process and fit the model
    :classifier: classifier for train
    :grid_param: different parameters of classifier to test
    :metrics: list of different metrics(e.g. accuracy,recall..)
    :x_all: features
    :y: labels
    """

    results={}

    X_train, X_test, y_train, y_test = train_test_split(x_all, y, test_size=0.33)

    for metric in metrics:

        gd_sr = GridSearchCV(estimator=classifier,
                        param_grid=grid_param,
                        scoring=metric,
                        cv=5,
                        n_jobs=-1)


        gd_sr.fit(X_train, y_train)
        
        results[str(metric)] = gd_sr.best_score_
   
    #Plot confusion matrix process
    y_pred = gd_sr.best_estimator_.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred) 
    print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))
    print(gd_sr.best_score_)
    print(conf_mat)
   
    np.set_printoptions(precision=2)

    plt.figure()
    plot_confusion_matrix(str(classifier), conf_mat, classes=set(y))
 
    return results    
    

def save_results(algorithm,results):
    """
    Print the results to files based on classifier  
    :param algorithm: name of the train algorithm
    :results: dictionary with results
    """
  
    for key,values in results.items():
        msg = "%s: %s---> %f" % (algorithm, key, values)
        print(msg,file = open(str(algorithm)+'_results.txt','a'))
    

def train_models(x, training_algorithms):
    """
    Check the name of given algorithm and train the proper model
    using Grid_Search_Process() then save results.
    :param x: features
    :training_algorithms: algorithm/s for training
    """

    for item in os.listdir():
        if item.endswith("results.txt"):   
           os.remove(item)

    x_all, y = data_preparation(x)

    metrics = ['accuracy','precision_macro','recall_macro','f1_macro']

    for algorithm in training_algorithms:
        
        if algorithm == 'SVM':
            classifier = SVC()
            grid_param = {
              'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}
            
            results = Grid_Search_Process(classifier, grid_param, metrics, x_all, y)
            
            save_results(algorithm, results)
            
        elif algorithm == 'Decision_Trees':
            classifier = DecisionTreeClassifier()
            grid_param = {
                'criterion': ['gini', 'entropy'],
                'max_depth':range(1, 10)}
            results = Grid_Search_Process(classifier, grid_param, metrics,
                                          x_all, y)
            
            save_results(algorithm, results)
           
        elif algorithm == 'KNN':
            classifier = KNeighborsClassifier()
            grid_param = {
                'n_neighbors': [3,5,7],
                'weights':['uniform','distance']}
            
            results = Grid_Search_Process(classifier, grid_param, metrics,
                                          x_all, y)
            
            save_results(algorithm, results)
        
        elif algorithm == 'Adaboost':
            classifier = AdaBoostClassifier()
            grid_param = {
                 'n_estimators': np.arange(100,250,50),
                 'learning_rate': [0.01, 0.05, 0.1, 1]}
            results = Grid_Search_Process(classifier, grid_param, metrics,
                                          x_all, y)
            
            save_results(algorithm, results)
        

        elif algorithm == 'Extratrees':
            classifier = ExtraTreesClassifier()
            grid_param = {
                'n_estimators': range(50,126,25),
                'max_features': range(50,401,50)}
            results = Grid_Search_Process(classifier, grid_param, metrics,
                                          x_all, y)
            
            save_results(algorithm, results)
        

        else:
            classifier = RandomForestClassifier()
            grid_param = {
            'n_estimators': [100, 300],
            'criterion': ['gini', 'entropy']}
            results = Grid_Search_Process(classifier, grid_param, metrics,
                                          x_all, y)
            
            save_results(algorithm, results)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    args = parse_arguments()
    videos_path = args.videos_path
    training_algorithms = args.training_algorithms

    
    # Convert list of lists to a single list
    videos_path = [item for sublist in videos_path for item in sublist]
    training_algorithms = [item for sublist in training_algorithms
                           for item in sublist]

    for paths in videos_path:
        assert os.path.exists(paths), "Video Path doesn't exist, " + \
                                    str(paths)

    available_algorithms = ['SVM', 'Decision_Trees', 'KNN', 'Adaboost',
                            'Extratrees', 'RandomForest']

    for algorithm in training_algorithms:
        if algorithm not in available_algorithms:
            sys.exit('%s is not available please read the '
                     'Usage example' % (algorithm))

    #Extract features of videos
    x, name_of_files = feature_extraction(videos_path)

    #Train the models
    train_models(x, training_algorithms)
    
