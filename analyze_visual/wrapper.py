"""
This script is used to predict video's class.
The input can be single video or directory. 

Usage example:

python3 wrapper.py -i dataset/Panoramic/trump.mp4 -m SVM -o output_file.csv

Available algorithms to use: SVM, Decision_Trees, KNN, Adaboost,
Extratrees, RandomForest
"""

import sys
import glob
import argparse
import os.path
import numpy as np
from numpy import unique
from numpy import where
import pandas as pd
from pickle import load
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

sys.path.insert(0, '..')
from analyze_visual.analyze_visual import process_video


def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Wrapper"
                                                 "Predict shot class")

    parser.add_argument("-i", "--input_videos_path",
                        action='append', nargs='+',
                        required=True, help="Videos folder path")

    parser.add_argument("-m", "--model", required=True, nargs=None,
                        help="Model")

    parser.add_argument("-o", "--output_file", required=True, nargs=None,
                        help="Output file with results")

    return parser.parse_args()


def create_dataframe(names):
    """
    Create an emtry dataframe with named columns
    :param name: names of classes
    :return: dataframe
    """
    # Create Dataframe
    names = names.tolist() 
    df = pd.DataFrame(columns=names)
    df.insert(0,'File_name', value = [])

    return df


def video_class_predict(features, algorithm):
    """
    Loads pre-trained model and predict single shot's class
    :param features: features
    :algorithm: Training algorithm
    :return probas: posterior probability of every class
    :classes: names of classes    
    """

    # Load the model
    model = load(open('shot_classifier_' + str(algorithm)+'.pkl', 'rb'))

    # Load the scaler
    scaler = load(open('shot_classifier_' + str(algorithm) +
                       '_scaler.pkl', 'rb'))

    # Transform the test dataset
    features_scaled = scaler.transform(features)

    # Predict the probabilities of every class
    classes = model.classes_
    results = model.predict_proba(features_scaled)

    # Convert list of lists to a single list
    probas = [item for sublist in results for item in sublist]

    return probas, classes

def clustering(videos_path, model, algorithm,
                               outfilename):
    """
    Clustering process
    :param videos_path: path to video directory of filename to be analyzed
    :param model: path name of the model
    :param algorithm: type of the modelling algorithm (e.g. SVM)
    :param outfilename: output csv filename (only for input folder)
    :return:
    """

    features_all= []
    final_df = create_dataframe(model.classes_)

    for movie in videos_path:

        f, c, ft, df = video_class_predict_folder(movie, model, algorithm,
                                outfilename)
        features_all.append(ft)
        final_df = final_df.append(df, ignore_index=True)

        print(f,c)
    
    final_df.to_csv(outfilename)

    #Convert list of lists to numpy array
    features = np.array(features_all)
    
    # Define scaler
    scaler = MinMaxScaler()

    # Fit scaler on the training dataset
    scaler.fit(features)

    # Transform both datasets
    scaled_features = scaler.transform(features)

    model = KMeans(n_clusters=len(videos_path))

    model.fit(scaled_features)

    yhat = model.predict(scaled_features)

    with open("cluster_prediction.txt", "w") as output:
        for movie,y in zip(videos_path,yhat):
            print(f'{movie} : {y}',file = output)
        
    clusters = unique(yhat)

    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        pyplot.scatter(scaled_features[row_ix, 0], scaled_features[row_ix, 1])

    # show the plot
    pyplot.savefig('plot_clusters.png')
    

def video_class_predict_folder(videos_path, model, algorithm,
                               outfilename):
    """
    video_class_predict_folder
    :param videos_path: path to video directory of filename to be analyzed
    :param model: path name of the model
    :param algorithm: type of the modelling algorithm (e.g. SVM)
    :param outfilename: output csv filename (only for input folder)
    :return:
    """
    features_all = []
    final_proba = np.empty((0, len(model.classes_)))
    df = create_dataframe(model.classes_)
    
    if os.path.exists(str(videos_path) + ".txt"):
        os.remove(str(videos_path) + ".txt")
    if os.path.isfile(videos_path):
        features_stats = process_video(videos_path, 2, True, True, True)
        features = features_stats[0]
        features = features.reshape(1, -1)
        # Predict the classes of shots
        probas, classes = video_class_predict(features, algorithm)
        for class_name, proba in zip(classes, probas):
            print(f'Video {videos_path} belongs by '
                  f'{proba} in {class_name} class')
        final_proba = probas

    elif os.path.isdir(videos_path):
        types = ('*.avi', '*.mpeg', '*.mpg', '*.mp4', '*.mkv', '*.webm')
        video_files_list = []
        for files in types:
            video_files_list.extend(glob.glob(os.path.join(videos_path, files)))
        video_files_list = sorted(video_files_list)
        for v in video_files_list:
            features_stats = process_video(v, 2, True, True, True)
            features = features_stats[0]
            features = features.reshape(1, -1)
            features = features.tolist()
            features_all.append(features)
            probas, classes = video_class_predict(features, algorithm)           
            # Save the resuls in a numpy array
            final_proba = np.append(final_proba, [probas], axis=0)
            # Convert format of file names
            splitting = v.split('/')
            v = splitting[-1]
            # Insert values to dataframe
            df = df.append({'File_name': v}, ignore_index=True)
        for i in range(2):
            features_all = [item for sublist in features_all for item in sublist]
        for i, class_name in enumerate(classes):
            df[class_name] = final_proba[:, i]
        # Save results to csv
        df.to_csv(outfilename)

        print(final_proba)
        final_proba = final_proba.mean(axis=0)
        # Print and save the final results
        with open(str(videos_path) + ".txt", "a") as text_file:
            for class_name, proba in zip(classes, final_proba):
                print(f'The movie {videos_path} belongs by '
                      f'{"{:.2%}".format(proba)} '
                      f'in {class_name} class', file=text_file)
                print(f'The movie {videos_path} '
                      f'belongs by {"{:.2%}".format(proba)} '
                      f'in {class_name} class')

    return final_proba, classes, features_all, df


def main():
    args = parse_arguments()
    videos_path = args.input_videos_path
    algorithm = args.model
    outfilename = args.output_file
    # Convert list of lists to a single list
    videos_path = [item for sublist in videos_path for item in sublist]    
    model = load(open('shot_classifier_' + str(algorithm)+'.pkl', 'rb'))

    if (len(videos_path)) > 1:    
        clustering(videos_path, model, algorithm, outfilename)
    else:
        videos_path = videos_path[-1]
        print(videos_path)
        f, c, _, _ = video_class_predict_folder(videos_path, model, algorithm,
                                    outfilename)
        print(f, c)

 

if __name__ == '__main__':
    main()
