"""
This script is used to train different ML algorithms and save the results. 
Usage example:
python3 train.py -v dataset/Aerial dataset/None -a SVM Decision_Trees
Available algorithms for traning: SVM, Decision_Trees, KNN, Adaboost,
Extratrees, RandomForest
"""

import warnings
from collections import deque
import argparse
import os
import numpy as np
import sys
import fnmatch
import itertools
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn import model_selection, preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, \
    RandomForestClassifier

sys.path.insert(0, '..')
from analyze_visual import dir_process_video
from sklearn.metrics import make_scorer, accuracy_score, precision_score, \
    recall_score, f1_score, plot_confusion_matrix, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split




from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold


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
    f_names = {}

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
                    f_names['f_name_{0}'.format(folder)] = np.load(os.path.join(folder,
                                                                     file))
        else:
            # calculate features for current folder:
            x["x_{0}".format(folder)],\
            name_of_files["paths_{0}".format(folder)],\
            f_names['f_name_{0}'.format(folder)]     = \
                dir_process_video(folder, 2, True, True, True)   

    return x, name_of_files, f_names

def plot_feature_histograms(list_of_feature_mtr, feature_names,
                            class_names, n_columns=5):
    '''
    Plots the histograms of all classes and features for a given
    classification task.
    :param list_of_feature_mtr: list of feature matrices
                                (n_samples x n_features) for each class
    :param feature_names:       list of feature names
    :param class_names:         list of class names, for each feature matr
    '''
    n_features = len(feature_names)
    n_bins = 12
    n_rows = int(n_features / n_columns) + 1
    figs = plotly.subplots.make_subplots(rows=n_rows, cols=n_columns,
                                      subplot_titles=feature_names)
    figs['layout'].update(height=(n_rows * 250))
    clr = get_color_combinations(len(class_names))

    for i in range(n_features):
        # for each feature get its bin range (min:(max-min)/n_bins:max)
        f = np.vstack([x[:, i:i + 1] for x in list_of_feature_mtr])
        bins = np.arange(f.min(), f.max(), (f.max() - f.min()) / n_bins)
        for fi, f in enumerate(list_of_feature_mtr):
            # load the color for the current class (fi)
            mark_prop = dict(color=clr[fi], line=dict(color=clr[fi], width=3))
            # compute the histogram of the current feature (i) and normalize:
            h, _ = np.histogram(f[:, i], bins=bins)
            h = h.astype(float) / h.sum()
            cbins = (bins[0:-1] + bins[1:]) / 2
            scatter_1 = go.Scatter(x=cbins, y=h, name=class_names[fi],
                                   marker=mark_prop, showlegend=(i == 0))
            # (show the legend only on the first line)
            figs.append_trace(scatter_1, int(i/n_columns)+1, i % n_columns+1)
    for i in figs['layout']['annotations']:
        i['font'] = dict(size=10, color='#224488')
    plotly.offline.plot(figs, filename="report.html", auto_open=True)

def get_color_combinations(n_classes):
    clr_map = plt.cm.get_cmap('jet')
    range_cl = range(int(int(255/n_classes)/2), 255, int(255/n_classes))
    clr = []
    for i in range(n_classes):
        clr.append('rgba({},{},{},{})'.format(clr_map(range_cl[i])[0],
                                              clr_map(range_cl[i])[1],
                                              clr_map(range_cl[i])[2],
                                              clr_map(range_cl[i])[3]))
    return clr

def data_preparation(x, f_name):
    """
    Prepare the data before the training process
    :param x: exracted features from videos
    :return: features and labels
    """
    x_all = np.empty((0, 244), float)
    f_names = np.empty((0, 244), float)
    y = []

    for key, value in x.items():
        x_all = np.append(x_all,value,axis=0)
        for i in range(value.shape[0]):
            y.append(str(key))

    for key, value in f_name.items():
        f_names = np.append(f_names,value,axis=0)

    # Standarization
    scaler = StandardScaler()
    # fit and transform the data
    x_all = scaler.fit_transform(x_all)
    print(x_all.shape)
    print(f_names.shape)
    print(f_names)
    print(x_all)
    

    
    #Fail try to plot features histogram
    
    #plot_feature_histograms(x_all, f_names, y)
    
    # Encode target labels with value between 0 and n_classes-1
    lb = preprocessing.LabelEncoder()
    y = lb.fit_transform(y)

    #Feature selection
    '''
    rfecv = RFECV(DecisionTreeClassifier(),cv=StratifiedKFold(10), step=1)
    rfecv = rfecv.fit(x_all, y)

    print('Optimal number of features: {}'.format(rfecv.n_features_))

    plt.figure(figsize=(16, 9))
    plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
    plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)

    plt.show()
    '''
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


def Grid_Search_Process(classifier, grid_param, x_all, y):
    """
    Hyperparameter tuning process and fit the model
    :classifier: classifier for train
    :grid_param: different parameters of classifier to test
    :x_all: features
    :y: labels
    """

    X_train, X_test, y_train, y_test = train_test_split(x_all, y, test_size=0.33)

    
    gd_sr = GridSearchCV(estimator=classifier,
                         param_grid=grid_param,
                         scoring='f1_macro',
                         cv=5,
                         n_jobs=-1)

    gd_sr.fit(X_train, y_train)
   
    # Plot confusion matrix process
    y_pred = gd_sr.best_estimator_.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred) 
    print(conf_mat)
    np.set_printoptions(precision=2)

    plt.figure()
    plot_confusion_matrix(str(classifier), conf_mat, classes=set(y))
 
    return y_test, y_pred   
    

def save_results(algorithm, y_test, y_pred):
    """
    Print the results to files based on classifier  
    :param algorithm: name of the train algorithm
    :y_test: values for test
    :y_pred: predicted values
    """
    results = {}
    results['accuracy'] = str(accuracy_score(y_test, y_pred))
    results['precision'] = str(precision_score(y_test, y_pred, average='macro'))
    results['recall'] = str(recall_score(y_test, y_pred, average='macro'))
    results['f1'] = str(f1_score(y_test, y_pred, average='macro'))

    for key, values in results.items():
        msg = "%s: %s---> %s" % (algorithm, key, values)
        print(msg, file=open(str(algorithm)+'_results.txt','a'))
    

def train_models(x, training_algorithms, f_names):
    """
    Check the name of given algorithm and train the proper model
    using Grid_Search_Process() then save results.
    :param x: features
    :param list of training_algorithms to use
    :training_algorithms: algorithm/s for training
    """

    for item in os.listdir():
        if item.endswith("results.txt"):   
           os.remove(item)

    x_all, y = data_preparation(x,f_names)

    for algorithm in training_algorithms:
        if algorithm == 'SVM':
            classifier = SVC()
            grid_param = {
              'C': [0.1, 0.5, 1, 5, 10, 100],
              'kernel': ['rbf']}
        elif algorithm == 'Decision_Trees':
            classifier = DecisionTreeClassifier()
            grid_param = {
                'criterion': ['gini', 'entropy'],
                'max_depth': range(1, 10)}
        elif algorithm == 'KNN':
            classifier = KNeighborsClassifier()
            grid_param = {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform','distance']}

        elif algorithm == 'Adaboost':
            classifier = AdaBoostClassifier()
            grid_param = {
                 'n_estimators': np.arange(100, 250, 50),
                 'learning_rate': [0.01, 0.05, 0.1, 1]}
        elif algorithm == 'Extratrees':
            classifier = ExtraTreesClassifier()
            grid_param = {
                'n_estimators': range(50, 126, 25),
                'max_features': range(50, 401, 50)}
        else:
            classifier = RandomForestClassifier()
            grid_param = {
            'n_estimators': [100, 300],
            'criterion': ['gini', 'entropy']}

        y_test,y_pred = Grid_Search_Process(classifier, grid_param, x_all, y)
        save_results(algorithm, y_test, y_pred)


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
    x, name_of_files, f_names = feature_extraction(videos_path)

    #Train the models
    train_models(x, training_algorithms, f_names)
