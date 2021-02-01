'''
This script is used to predict video's class. 

Usage example:

Run single file:

python3 wrapper.py -f dataset/Panoramic/trump.mp4 -m SVM

Run directory:

python3 wrapper.py -d dataset/Panoramic -m SVM

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
from analyze_visual.analyze_visual import process_video
from train import feature_extraction, data_preparation

def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Wrapper"
                                                 "Predict shot class")

    parser.add_argument("-f", "-d", "--videos_path", required=True, action='append',
                        nargs='+', help="Videos folder path")

    parser.add_argument("-m", "--model", required=True, nargs=None,
                        help="Model")

    return parser.parse_args()


def video_class_predict(X_test,y_test,algorithm):
    """
    Loads pre-trained model and predict single shot's class
    :param features: features
    :labels: labels
    :algorithm: Training algorithm
    :return:    
    """

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


def dir_class_predict(features,labels,algorithm):
    """
    Loads pre-trained model and predict videos class
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


def main(argv):

    args = parse_arguments()
    videos_path = args.videos_path
    algorithm = args.model

    # Convert list of lists to a single list
    videos_path = [item for sublist in videos_path for item in sublist]
    
    if argv[1] == "-f":
        
        shot = videos_path[0]

        #Extract features of video and save features and label
        features_stats = process_video(shot, 2, True, True, True)
        features = features_stats[0]
        features = features.reshape(1,-1)

        labels = ['Handled' for i in range(features.shape[0])]

        #Predict the classes of shots
        video_class_predict(features,labels,algorithm)


    else:

        #Extract features of videos
        x, _, _ = feature_extraction(videos_path)

        #Prepare the data    
        features,labels = data_preparation(x)

        #Predict the classes of shots
        dir_class_predict(features,labels,algorithm)

    
if __name__ == '__main__':
    main(sys.argv)
    

    