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

    parser.add_argument("-i", "--input_videos_path",
                        required=True,
                        nargs=None,
                        help="Videos folder path")

    parser.add_argument("-m", "--model", required=True, nargs=None,
                        help="Model")

    return parser.parse_args()


def video_class_predict(X_test,algorithm):
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
    
    return results


def main(argv):

    args = parse_arguments()
    videos_path = args.input_videos_path
    algorithm = args.model


    if os.path.isfile(videos_path):
        features_stats = process_video(videos_path, 2, True, True, True)
        features = features_stats[0]
        features = features.reshape(1, -1)
        #Predict the classes of shots
        r = video_class_predict(features, algorithm)
        print(f'Video {videos_path} belongs to {r}')
    elif os.path.isdir(videos_path):
        import glob
        types = ('*.avi', '*.mpeg', '*.mpg', '*.mp4', '*.mkv', '*.webm')
        video_files_list = []
        for files in types:
            video_files_list.extend(glob.glob(os.path.join(videos_path, files)))
        video_files_list = sorted(video_files_list)
        for v in video_files_list:
            features_stats = process_video(v, 2, True, True, True)
            features = features_stats[0]
            features = features.reshape(1, -1)
            # Predict the classes of shots
            r = video_class_predict(features, algorithm)
            print(f'Video {v} belongs to {r}')
    
if __name__ == '__main__':
    main(sys.argv)
    

    