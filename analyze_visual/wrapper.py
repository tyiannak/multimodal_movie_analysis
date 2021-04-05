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

    parser.add_argument("-c", "--num_of_clusters", required=False, nargs=None,
                        default=2, help="number of clusters used"
                                        "(clustering is executed only if "
                                        "number of directories is more than 1")

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
        print(video_files_list)
        count_processed = 0
        for v in video_files_list:
            try:
                # TODO remove long try-except statement
                print(algorithm)
                print(v)
                features_stats = process_video(v, 2, True, True, True)
                features = features_stats[0]
                features = features.reshape(1, -1)
                probas, classes = video_class_predict(features, algorithm)           
                # Save the resuls in a numpy array
                final_proba = np.append(final_proba, [probas], axis=0)
                # Convert format of file names
                splitting = v.split(os.sep)
                v = splitting[-1]
                # Insert values to dataframe
                df = df.append({'File_name': v}, ignore_index=True)
                count_processed += 1
            except:
                print('This video is corrupted')
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

    return final_proba, classes, df


def clustering(videos_path, model, algorithm, outfilename, nclusters=2):
    """
    Clustering process
    :param videos_path: path to video directory of filename to be analyzed
    :param model: path name of the model
    :param algorithm: type of the modelling algorithm (e.g. SVM)
    :param outfilename: output csv filename (only for input folder)
    :param nclusters: number of clusters to use
    :return:
    """
    final_proba = []
    final_df = create_dataframe(model.classes_)
   
    for movie in videos_path:

        f, c, df = video_class_predict_folder(movie, model, algorithm,
                                              outfilename)

        final_proba.append(f)
        final_df = final_df.append(df, ignore_index=True)
        print(f, c)

    final_df.to_csv(outfilename)
    final_proba = np.array(final_proba)
    print(final_proba)
    model = KMeans(n_clusters=nclusters)
    model.fit(final_proba)
    yhat = model.predict(final_proba)
    with open("cluster_prediction.txt", "w") as output:
        for movie, y in zip(videos_path,yhat):
            print(f'{movie} : {y}',file = output)
        
    clusters = unique(yhat)
    print(clusters)
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        pyplot.scatter(final_proba[row_ix, 0], final_proba[row_ix, 1])

    # show the plot
    pyplot.savefig('plot_clusters.png')


def main():
    args = parse_arguments()
    videos_path = args.input_videos_path
    algorithm = args.model
    outfilename = args.output_file
    nclusters = int(args.num_of_clusters)
    # Convert list of lists to a single list
    videos_path = [item for sublist in videos_path for item in sublist]    
    model = load(open('shot_classifier_' + str(algorithm)+'.pkl', 'rb'))

    if (len(videos_path)) > 1:    
        clustering(videos_path, model, algorithm, outfilename, nclusters)
    else:
        videos_path = videos_path[-1]
        print(videos_path)
        f, c, _ = video_class_predict_folder(videos_path, model, algorithm,
                                             outfilename)
        print(f, c)

 

if __name__ == '__main__':
    main()
