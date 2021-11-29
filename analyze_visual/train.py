"""
This script is used to train different ML algorithms and save the results. 

Usage example:

python3 train.py -v dataset/Aerial dataset/None -a SVM Decision_Trees

Available algorithms for traning: SVM, Decision_Trees, KNN, Adaboost,
Extratrees, RandomForest
"""


import warnings
import argparse
import shutil
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
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, \
    RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score, plot_confusion_matrix, confusion_matrix, \
    roc_curve, auc, precision_recall_curve, average_precision_score    
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
import json
sys.path.insert(0, '..')
from analyze_visual.analyze_visual import dir_process_video


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

def prep_roc_pr_curves(gd_sr, y_score, y_test):
    """
    Prepate data and plot roc and precision-recall curves.
    :param gd_sr: grid search
    :y_score: predicted labels
    :y_test: labels
    """
    #Create directories to save plots
    path = 'Roc_curves'
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    path = 'Precision-Recall_curves'
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    #Save the number of classes
    classes_names = list(set(y_test))
    n_classes = len(classes_names)

    y_pos_label = []

    #One hot encoding
    p = 0
    for name in gd_sr.classes_:
        for i in y_test:
            if i == name:
                y_pos_label.append(1)
            else:
                y_pos_label.append(0)
        #Plot roc curves
        plot_roc_curve(y_score[:,p], y_pos_label, n_classes, name)
        #Plot Precision-Recall curves
        prec_rec_curve(y_score[:,p], y_pos_label, n_classes, name)  
        
        y_pos_label.clear()
        p+=1

def plot_confusion_matrix(name, cm, classes):
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
    plt.savefig("shot_classifier_conf_mat_" + str(name) + ".jpg")

def plot_roc_curve(y_score, y_test, n_classes, name):
    """
    Plot ROC curve
    :y_score: Predicted labels
    :y_test: labels of test set 
    :n_classes: Number of classes   
    """
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot of a ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc curve for '+str(name)+' class')
    plt.legend(loc="lower right")
    plt.savefig('Roc_curves/Roc_curve_'+str(name)+'.png')

def prec_rec_curve(y_score, y_test, n_classes, name):
    """
    Plot Precision-Recall curve
    :y_score: Predicted labels
    :y_test: labels of set y
    :n_classes: Number of classes   
    """
    # Compute Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    #Plot PRecision-Recall curve
    plt.figure()
    plt.step(recall, precision, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve for '+str(name)+' class')

    plt.savefig('Precision-Recall_curves/Precision-Recall-curve_'+str(name)+'.png')

def Grid_Search_Process(classifier, grid_param, x_all, y):
    """
    Hyperparameter tuning process and fit the model
    :classifier: classifier for train
    :grid_param: different parameters of classifier to test
    :x_all: features
    :y: labels
    """

    X_train, X_test, y_train, y_test = train_test_split(x_all, y,
                                                        test_size=0.33)
    print(X_train.shape, X_test.shape)
    from imblearn.over_sampling import SMOTE
    smt = SMOTE(random_state=0)
    X_train, y_train = smt.fit_resample(X_train, y_train)
    print(X_train.shape, X_test.shape)

    # Define scaler
    scaler = MinMaxScaler()

    # Fit scaler on the training dataset
    scaler.fit(X_train)

    # Transform both datasets
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    gd_sr = GridSearchCV(estimator=classifier,
                         param_grid=grid_param,
                         scoring='f1_macro',
                         cv=5, n_jobs=-1)

    gd_sr.fit(X_train_scaled, y_train)
   
    # Save model    
    dump(gd_sr, open('shot_classifier_' + str(algorithm)+'.pkl', 'wb'))

    # Save the scaler
    dump(scaler, open('shot_classifier_' + str(algorithm) +
                      '_scaler.pkl', 'wb'))


    class_labels = list(set(y_test))
    # Plot confusion matrix process
    y_pred = gd_sr.best_estimator_.predict(X_test_scaled)
    conf_mat = confusion_matrix(y_test, y_pred, labels=class_labels)
    print(conf_mat)
    np.set_printoptions(precision=2)

    plot_confusion_matrix(str(classifier), conf_mat, classes=class_labels)

    y_pred_proba = gd_sr.best_estimator_.predict_proba(X_test_scaled)

    #Prepare data and plot roc, precision-recall cruves
    prep_roc_pr_curves(gd_sr, y_pred_proba, y_test)
 
    return y_test, y_pred   
    

def save_results(algorithm, y_test, y_pred):
    """
    Print the results to files based on classifier  
    :param algorithm: name of the train algorithm
    :y_test: values for test
    :y_pred: predicted values
    """
    results = {}
    class_names = list(set(y_test))
    precisions = precision_score(y_test, y_pred, average=None,
                                 labels=class_names)
    recalls = recall_score(y_test, y_pred, average=None,
                              labels=class_names)

    results['accuracy'] = accuracy_score(y_test, y_pred)
    results['f1'] = str(f1_score(y_test, y_pred, average='macro'))
    results['precision_mean'] = precision_score(y_test, y_pred, average='macro')
    results['recall_mean'] = recall_score(y_test, y_pred, average='macro')
    results['precisions'] = {class_names[i]: precisions[i]
                             for i in range(len(class_names))}
    results['recalls'] = {class_names[i]: recalls[i]
                          for i in range(len(class_names))}
    with open("shot_classifier_" + algorithm + "_results.json", 'w') as fp:
        json.dump(results, fp, indent=4)


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
            classifier = SVC(probability=True)
            grid_param = {
              'C': [0.1, 0.5, 1, 2, 5, 10, 100],
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
                'n_estimators': range(25, 126, 25),
                'max_features': range(25, 401, 25)}
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
                     'Usage example' % algorithm)

    # Extract features of videos
    x, name_of_files, _ = feature_extraction(videos_path)

    # Train the models
    train_models(x, training_algorithms)

