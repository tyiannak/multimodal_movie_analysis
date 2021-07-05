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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
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


def cloning(original_list):
    """cloning a list"""
    list_copy = original_list[:]

    return list_copy


def accuracy(confusion_matrix):
    """
    calculates average accuracy for
    aggregated confusion matrix.
    """
    diag_sum = confusion_matrix.trace()
    total_sum = confusion_matrix.sum()
    avg_acc = diag_sum / total_sum

    return avg_acc


def f1_macro_avg(confusion_matrix):
    """
    Calculates macro-averaged f1-score
    from aggregated confusion matrix
    """

    conf_mat_rows = len(confusion_matrix)
    conf_mat_cols = len(confusion_matrix[0])

    precision_tmp = 0
    recall_tmp = 0
    precision = []
    recall = []
    for i in range(conf_mat_rows):
        for j in range(conf_mat_cols):
            precision_tmp = precision_tmp + confusion_matrix[i][j]
        precision_tmp = confusion_matrix[i][i] / precision_tmp
        precision.append(precision_tmp)
        precision_tmp = 0

    for j in range(conf_mat_cols):
        for i in range(conf_mat_rows):
            recall_tmp = recall_tmp + confusion_matrix[i][j]
        recall_tmp = confusion_matrix[j][j] / recall_tmp
        recall.append(recall_tmp)
        recall_tmp = 0

    #print("precisions: ", precision)
    #print("recalls: ", recall)

    f1_scr = []
    for prec_i, rec_i in zip(precision, recall):
        f1_scr_tmp = 2 * (prec_i * rec_i) / (prec_i + rec_i)
        f1_scr.append(f1_scr_tmp)

    f1_macro = sum(f1_scr) / len(f1_scr)

    return f1_macro


def create_windows(videos_path):
    """
    Checks if windowing is done otherwise it creates windows
    (using split_feature_matrix() function)
    :param videos_path: directory of videos
    """

    for folder in videos_path:
        # for each class-folder check if windowing is done since
        # a .npy file is created as soon as the windowing process is done
        window_file_exists = fnmatch.filter(os.listdir(folder),
                                            str(int(window_step*process_step)) + '*_sec_window_done.npy')

        if len(window_file_exists) <= 0:
            # if windowing is not done
            split_feature_matrix(folder)


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
        plt.savefig("shot_classifier_conf_mat_" + str(name) + ".jpg")
    elif id == 1:
        plt.savefig("shot_classifier_conf_mat_window_" + str(name) + ".jpg")


def window_grid_search(videos_path, classifier, grid_param):
    """
    Hyperparameter tuning process and fit the model
    (for windowing process).
    :videos_path: directory of videos
    :classifier: classifier for train
    :grid_param: different parameters of classifier to test
    """

    x_extended = []

    for folder in videos_path:
        # get a list of the (full path) names of the *_extended.npy files for all class-folders
        # (where features per window size are saved)
        np_extended = fnmatch.filter(os.listdir(folder), '*_extended.npy')
        for filename in np_extended:
            full_path_name = folder+"/"+filename
            x_extended.append(full_path_name)

    y_extended = cloning(x_extended)

    # create y label
    for i, label in enumerate(y_extended):
        splitting = label.split('/')
        label = splitting[-2]
        y_extended[i] = label

    x_array = np.array([x_extended])
    y_array = np.array([y_extended])

    # new_array is a 2D array in which the first row
    # contains the names of the _extented.npy files
    # and the second one contains the corresponding y labels
    new_array = np.vstack((x_array, y_array)).T

    #shuffle data
    #np.random.shuffle(new_array)

    # with open("extended_f_names.csv", "w+") as to_csv:
    #     csvWriter = csv.writer(to_csv, delimiter=',')
    #     csvWriter.writerows(new_array)

    X = new_array[:, 0]
    y = new_array[:, 1]

    smt = SMOTE(random_state=0)
    scaler = MinMaxScaler()

    # create pipline
    pipe = Pipeline(steps=[('smote', smt), ('scaler', scaler), ('clf', classifier)])

    splits = 5
    repeats = 10
    rskf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats)
    # skf = StratifiedKFold(n_splits=5)

    gd_sr = GridSearchCV(pipe,
                         param_grid=grid_param,
                         scoring='f1_macro',
                         cv=rskf, n_jobs=-1)

    counter = 0
    acc = 0
    f1_counter = 0

    num_of_folds = 0
    print("\nRepeated Stratified KFold for windowing process has started! Please wait... \n")
    for train_index, test_index in gd_sr.cv.split(X, y):
        num_of_folds = num_of_folds + 1
        print("fold: ", num_of_folds)

        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train_arr, X_test_arr = X[train_index], X[test_index]
        y_train_arr, y_test_arr = y[train_index], y[test_index]

        X_train = np.empty((0, 348), float)
        X_test = np.empty((0, 348), float)
        y_train = []
        y_test = []

        # create X_train, y_train in such a way
        # that X_train contains feature matrices which
        # correspond to a whole video
        for i, j in zip(X_train_arr, y_train_arr):
            np_skf_train = np.load(i)
            X_train = np.vstack((X_train, np_skf_train))
            rows = np_skf_train.shape[0]
            l = [j] * rows
            y_train.extend(l)

        # create X_test, y_test in such a way
        # that X_test contains feature matrices which
        # correspond to a whole video
        for i, j in zip(X_test_arr, y_test_arr):
            np_skf_test = np.load(i)
            X_test = np.vstack((X_test, np_skf_test))
            rows = np_skf_test.shape[0]
            k = [j] * rows
            y_test.extend(k)

        X_train = np.asmatrix(X_train)
        X_test = np.asmatrix(X_test)

        gd_sr.fit(X_train, y_train)
        class_labels = list(set(y_test))

        # create aggregated confusion matrix process
        y_pred = gd_sr.best_estimator_.predict(X_test)
        conf_mat = confusion_matrix(y_test, y_pred, labels=class_labels)

        rows = len(conf_mat)
        cols = len(conf_mat[0])

        if counter == 0:
            conf_mat_aggr = np.zeros((rows, cols))
            counter = -1

        for i in range(len(conf_mat)):
            for j in range(len(conf_mat[0])):
                conf_mat_aggr[i][j] = conf_mat_aggr[i][j] + conf_mat[i][j]

        # acc = acc + accuracy_score(y_test, y_pred)
        # f1_counter = f1_counter + f1_score(y_test, y_pred, average='macro')


    conf_mat_aggr = conf_mat_aggr.astype(int)
    print("\nAggregated Confusion Matrix:\n", conf_mat_aggr, "\n")

    # avg_acc = acc / (splits * repeats)
    # avg_f1 = f1_counter / (splits * repeats)

    avg_acc = accuracy(conf_mat_aggr)
    print("Average accuracy (window process): ", "{:.2f}".format(avg_acc * 100), "%")

    avg_f1 = f1_macro_avg(conf_mat_aggr)
    print("Average f1 score (window process): ", "{:.2f}".format(avg_f1 * 100), "%")

    win_id = 1
    np.set_printoptions(precision=2)
    plot_confusion_matrix(str(classifier), conf_mat_aggr, classes=class_labels, id=win_id)


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

    # create pipeline
    pipe = Pipeline(steps=[('smote', smt), ('scaler', scaler), ('clf', classifier)])

    splits = 5
    repeats = 10
    rskf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats)
    #skf = StratifiedKFold(n_splits=5)

    gd_sr = GridSearchCV(pipe,
                         param_grid=grid_param,
                         scoring='f1_macro',
                         cv=rskf, n_jobs=-1)

    counter = 0
    acc = 0
    f1_counter = 0
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

        # create aggregated confusion matrix process
        y_pred = gd_sr.best_estimator_.predict(X_test)
        conf_mat = confusion_matrix(y_test, y_pred, labels=class_labels)

        rows = len(conf_mat)
        cols = len(conf_mat[0])

        if(counter == 0):
            conf_mat_aggr = np.zeros((rows, cols))
            counter = -1

        for i in range(len(conf_mat)):
            for j in range(len(conf_mat[0])):
                conf_mat_aggr[i][j] = conf_mat_aggr[i][j] + conf_mat[i][j]

        # acc = acc + accuracy_score(y_test, y_pred)
        # f1_counter = f1_counter + f1_score(y_test, y_pred, average='macro')

    conf_mat_aggr = conf_mat_aggr.astype(int)
    print("\nAggregated Confusion Matrix:\n", conf_mat_aggr, "\n")

    # calculate average acc
    #avg_acc = acc / (splits * repeats)
    #avg_f1 = f1_counter / (splits * repeats)

    avg_acc = accuracy(conf_mat_aggr)
    print("Average accuracy: ", "{:.2f}".format(avg_acc * 100), "%")

    avg_f1 = f1_macro_avg(conf_mat_aggr)
    print("Average f1 score: ", "{:.2f}".format(avg_f1 * 100), "%")

    # Save the model
    dump(gd_sr, open('shot_classifier_' + str(algorithm) +
                     '.pkl', 'wb'))

    # Save the pipeline
    dump(pipe, open('shot_classifier_' + str(algorithm) +
                      '_pipeline.pkl', 'wb'))

    agg_id = 0
    np.set_printoptions(precision=2)
    plot_confusion_matrix(str(classifier), conf_mat_aggr, classes=class_labels, id=agg_id)
    

# def save_results(algorithm, y_test, y_pred):
#     """
#     Print the results to files based on classifier
#     :param algorithm: name of the train algorithm
#     :y_test: values for test
#     :y_pred: predicted values
#     """
#     results = {}
#     class_names = list(set(y_test))
#     precisions = precision_score(y_test, y_pred, average=None,
#                                  labels=class_names)
#     recalls = recall_score(y_test, y_pred, average=None,
#                               labels=class_names)
#
#     results['accuracy'] = accuracy_score(y_test, y_pred)
#     results['f1'] = str(f1_score(y_test, y_pred, average='macro'))
#     results['precision_mean'] = precision_score(y_test, y_pred, average='macro')
#     results['recall_mean'] = recall_score(y_test, y_pred, average='macro')
#     results['precisions'] = {class_names[i]: precisions[i]
#                              for i in range(len(class_names))}
#     results['recalls'] = {class_names[i]: recalls[i]
#                           for i in range(len(class_names))}
#     with open("shot_classifier_" + algorithm + "_results.json", 'w') as fp:
#         json.dump(results, fp, indent=4)


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
                'n_estimators': range(25, 126, 25),
                'max_features': range(25, 401, 25)}
        else:
            classifier = RandomForestClassifier()
            grid_param = {
            'n_estimators': [100, 300],
            'criterion': ['gini', 'entropy']}

        # y_test,y_pred = Grid_Search_Process(classifier, grid_param, x_all, y)
        # save_results(algorithm, y_test, y_pred)

        # Grid Search Process for aggregated features
        grid_search_process(classifier, grid_param, x_all, y)

        # Windowing Process
        create_windows(videos_path)
        window_grid_search(videos_path, classifier, grid_param)


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

