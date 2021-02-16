"""
This script is used to plot ROC curve. 

Usage example:

python3 train.py -v dataset/Aerial dataset/Static
"""
import os
import argparse
import numpy as np
from scipy import interp
from itertools import cycle
from train import feature_extraction,data_preparation,remove_features
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn import preprocessing



def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Plot Roc curves ")
    parser.add_argument("-v", "--videos_path", required=True, action='append',
                        nargs='+', help="Videos folder path")    

    return parser.parse_args()

def plot_roc_curve(y_score, y_test, n_classes):
    """
    Plot ROC curve
    :y_score: Predicted labels
    :y_test: labels of test y 
    :n_classes: Number of classes   
    """
    path = 'Roc_curves'
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    for i in range(n_classes):
        plt.figure()
        plt.plot(fpr[i], tpr[i],color='darkorange',
                 label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for ExtraTrees classifier')
        plt.legend(loc="lower right")
        plt.savefig('Roc_curves/Roc_curve_class_'+str(i)+'.png')

def prec_rec_curve(y_score, y_test, n_classes):
    """
    Plot Precision-Recall curve
    :y_score: Predicted labels
    :y_test: labels of test y 
    :n_classes: Number of classes   
    """
    path = 'Precision-Recall_curves'
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
        y_score.ravel())
    for i in range(n_classes):
        plt.figure()
        plt.step(recall[i], precision[i], where='post')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve')

        plt.savefig('Precision-Recall_curves/Precision-Recall-curve_class_'+str(i)+'.png')

def train_model(x_all, y):
    """
    Train the model
    :x_all: features
    :y: Labels
    """
    lb = preprocessing.LabelBinarizer()
    y= lb.fit_transform(y)
 
    n_classes = 8

    X_train, X_test, y_train, y_test = train_test_split(x_all, y,
                                                        test_size=0.33)

    # Define scaler
    scaler = MinMaxScaler()

    # Fit scaler on the training dataset
    scaler.fit(X_train)

    # Transform both datasets
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
      
    classifier = OneVsRestClassifier(ExtraTreesClassifier())

    grid_param = {
        'estimator__n_estimators': range(25, 126, 25),
        'estimator__max_features': range(25, 401, 25)}

    # Grid search process
    gd_sr = GridSearchCV(estimator=classifier,
                         param_grid=grid_param,
                         scoring='f1_macro',
                         cv=5, n_jobs=-1)

    y_score = gd_sr.fit(X_train_scaled,y_train).predict_proba(X_test_scaled)


    #Plot roc curves
    plot_roc_curve(y_score, y_test, n_classes)

    #Plot Precision-Recall curves
    prec_rec_curve(y_score, y_test, n_classes)
  
if __name__ == "__main__": 

    args = parse_arguments()
    videos_path = args.videos_path

    # Convert list of lists to a single list
    videos_path = [item for sublist in videos_path for item in sublist]

    #Feature extraction process
    x, _, _ = feature_extraction(videos_path)

    #Prepare data for plot
    features, labels = data_preparation(x)

    #Plot features histogram   
    train_model(features,labels)
