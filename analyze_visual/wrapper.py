'''
This script is used to predict video's class. 

Usage example:

python3 wrapper.py -f dataset/trump.mp4 -m SVM

Available algorithms to use: SVM, Decision_Trees, KNN, Adaboost,
Extratrees, RandomForest
'''
import sys
import pickle
import argparse
import os.path
from os import path
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pickle import load
sys.path.insert(0, '..')
from train import feature_extraction, data_preparation

def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Wrapper"
                                                 "Predict shot class")

    parser.add_argument("-v", "--videos_path", required=True, action='append',
                        nargs='+', help="Videos folder path")

    parser.add_argument("-m", "--model", required=True, nargs=None,
                        help="Model")

    return parser.parse_args()

def model_predict(features,labels,algorithm):
    """
    Loads pre-trained model and predict shot's class
    :param features: features
    :labels: labels
    :algorithm: Training algorithm
    :return:
    """

    _, X_test, _, y_test = train_test_split(features, labels, test_size=0.33)

    # Load the model
    model = load(open('trained_'+str(algorithm)+'.pkl', 'rb'))

    # Load the scaler
    scaler = load(open(str(algorithm)+'_scaler.pkl', 'rb'))

    # Transform the test dataset
    X_test_scaled = scaler.transform(X_test)

    # Predict the class
    results = model.predict(X_test_scaled)
    
    #Print the results
    acc = accuracy_score(y_test, results)
    print('Test Accuracy of classifier: ', acc)
    print('The shots belongs to class: ', results)


if __name__ == "__main__":

    args = parse_arguments()
    videos_path = args.videos_path
    algorithm = args.model

    # Convert list of lists to a single list
    videos_path = [item for sublist in videos_path for item in sublist]

    #Extract features of videos
    x, _, _ = feature_extraction(videos_path)

    #Prepare the data    
    features,labels = data_preparation(x)

    #Predict the classes of shots
    model_predict(features,labels,algorithm)



    

    