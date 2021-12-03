"""
This script is used to train different ML algorithms and save the results. 

Usage example:

python3 train.py -v dataset/Aerial dataset/None -a SVM Decision_Trees

Available algorithms for traning: SVM, Decision_Trees, KNN, Adaboost,
Extratrees, RandomForest
"""

import warnings
import argparse
import os
import numpy as np
import sys
import fnmatch
import itertools
from pickle import dump
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, \
    RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score, plot_confusion_matrix, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
import json
sys.path.insert(0, '..')
from analyze_visual.analyze_visual import *


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

    for folder in videos_path: # for each class-folder
        # get list of np files in that folder (where features can have
        # been saved):
        np_feature_files = fnmatch.filter(os.listdir(folder), '*_features.npy')
        np_fnames_files = fnmatch.filter(os.listdir(folder), '*_f_names.npy')

        print(np_feature_files)
        print(np_fnames_files)

        # if feature npy files exist:
        if len(np_feature_files) > 0 and len(np_fnames_files) > 0:
            for file in np_feature_files:
                if os.path.isfile(os.path.join(folder, file)):
                    x["{0}".format(folder)] = np.load(os.path.join(folder,
                                                                   file))
            for file in np_fnames_files:
                if os.path.isfile(os.path.join(folder, file)):
                    f_names['f_name_{0}'.format(folder)] = np.load(
                        os.path.join(folder,
                                     file))
        else:
            # calculate features for current folder:
            x["x_{0}".format(folder)], \
            name_of_files["paths_{0}".format(folder)], \
            f_names['f_name_{0}'.format(folder)] = \
                dir_process_video(folder, 2, True, True, True)

    return x, name_of_files, f_names


def data_preparation(x):
    """
    Prepare the data before the training process
    :param x: exracted features from videos
    :return: features and labels
    """
    y = []
    count_classes = 0
    for key, value in x.items():
        if count_classes == 0:
            x_all = value
        else:
            x_all = np.append(x_all, value, axis=0)
        count_classes += 1
        for i in range(value.shape[0]):
            y.append(str(key))

    # Convert format of labels
    for i, label in enumerate(y):
        splitting = label.split('/')
        label = splitting[-1]
        y[i] = label

    return x_all, y


def plot_confusion_matrix(name, cm, classes, id):
    """
    Plot confusion matrix
    :name: name of classifier
    :cm: estimates of confusion matrix
    :classes: all the classes
    """
    plt.figure()
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
    if id == 0:
        #plt.savefig(len(videos_path) + "_shot_classifier_conf_mat_" + str(name) + ".jpg")
        plt.savefig(str(len(videos_path)) + '_shot_classifier_conf_mat_' + str(name) + '.eps', format='eps')
    elif id == 1:
        plt.savefig("shot_classifier_conf_mat_window_" + str(name) + ".jpg")


def grid_search_process(classifier, grid_param, x_all, y):
    """
    Hyperparameter tuning process and fit the model
    (for aggregated features).
    :classifier: classifier for train
    :grid_param: different parameters of classifier to test
    :x_all: features
    :y: labels
    """

    smt = SMOTE(random_state=0)
    scaler = MinMaxScaler()
    #scaler = StandardScaler()

    # create pipeline
    pipe = Pipeline(steps=[('smote', smt), ('scaler', scaler), ('clf', classifier)])

    splits = 5
    repeats = 10
    rskf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats)

    gd_sr = GridSearchCV(pipe, param_grid=grid_param, scoring='f1_macro',
                         cv=rskf, n_jobs=-1)

    y_test_aggr = []
    y_pred_aggr = []

    print("\nRepeated Stratified KFold (for aggregated features) has started! Please wait... \n")
    for train, test in gd_sr.cv.split(x_all, y):
        X_train, X_test = x_all[train], x_all[test]
        X_train = np.asmatrix(X_train)
        X_test = np.asmatrix(X_test)

        y_train = []
        y_test = []

        for i in train:
            y_train.append(y[i])

        for j in test:
            y_test.append(y[j])

        gd_sr.fit(X_train, y_train)
        class_labels = list(set(y_test))

        y_pred = gd_sr.best_estimator_.predict(X_test)

        y_test_aggr.append(y_test)
        y_pred_aggr.append(y_pred)

    # calculate aggregated metrics process
    y_test_aggr_flattened = np.concatenate(y_test_aggr).ravel()
    y_pred_aggr_flattened = np.concatenate(y_pred_aggr).ravel()

    cm = confusion_matrix(y_test_aggr_flattened, y_pred_aggr_flattened)
    f1_score_macro = f1_score(y_test_aggr_flattened, y_pred_aggr_flattened, average='macro')
    acc = accuracy_score(y_test_aggr_flattened, y_pred_aggr_flattened)
    precision_recall_fscore = precision_recall_fscore_support(y_test_aggr_flattened, y_pred_aggr_flattened, average='macro')

    print("\nClassification Report:\n"
          "accuracy: {:0.2f}%,".format(acc * 100),
          "precision: {:0.2f}%,".format(precision_recall_fscore[0] * 100),
          "recall: {:0.2f}%,".format(precision_recall_fscore[1] * 100),
          "f1_score (macro): {:0.2f}%".format(f1_score_macro * 100))
    print("\nConfusion matrix\n", cm)

    agg_id = 0
    np.set_printoptions(precision=2)
    plot_confusion_matrix(str(classifier), cm, classes=class_labels, id=agg_id)

    #num_of_classes = len(videos_path)
    #np.save(str(num_of_classes) + "_class_y_test.npy", y_test_aggr_flattened)
    #np.save(str(num_of_classes) + "_class_y_pred.npy", y_pred_aggr_flattened)


def train_models(x, training_algorithms):
    """
    Check the name of given algorithm and train the proper model
    using Grid_Search_Process() then save results.
    :param x: features
    :param training_algorithms: list of training_algorithms to use
    :training_algorithms: algorithm/s for training
    """

    x_all, y = data_preparation(x)

    for algorithm in training_algorithms:
        if algorithm == 'SVM':
            classifier = SVC()
            grid_param = {
              'clf__C': [0.1, 0.5, 1, 2, 5, 10, 100],
              'clf__kernel': ['rbf']}
        elif algorithm == 'Decision_Trees':
            classifier = DecisionTreeClassifier()
            grid_param = {
                'clf__criterion': ['gini', 'entropy'],
                'clf__max_depth': range(1, 10)}
        elif algorithm == 'KNN':
            classifier = KNeighborsClassifier()
            grid_param = {
                'clf__n_neighbors': [3, 5, 7],
                'clf__weights': ['uniform','distance']}

        elif algorithm == 'Adaboost':
            classifier = AdaBoostClassifier()
            grid_param = {
                 'clf__n_estimators': np.arange(100, 250, 50),
                 'clf__learning_rate': [0.01, 0.05, 0.1, 1]}
        elif algorithm == 'Extratrees':
            classifier = ExtraTreesClassifier()
            grid_param = {
                'clf__n_estimators': range(25, 126, 25),
                'clf__max_features': range(25, 401, 25)}
        else:
            classifier = RandomForestClassifier()
            grid_param = {
            'clf__n_estimators': [100, 300],
            'clf__criterion': ['gini', 'entropy']}

        # Grid Search Process for aggregated features
        grid_search_process(classifier, grid_param, x_all, y)


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
                     'Usage example' % algorithm)

    # Extract features of videos
    x, name_of_files, _ = feature_extraction(videos_path)

    # Train the models
    train_models(x, training_algorithms)

