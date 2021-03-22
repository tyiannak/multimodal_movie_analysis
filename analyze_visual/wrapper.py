"""
This script is used to predict video's class.
The input can be single video or directory. 

Usage example:

python3 wrapper.py -i dataset/Panoramic/trump.mp4 -m SVM

Available algorithms to use: SVM, Decision_Trees, KNN, Adaboost,
Extratrees, RandomForest
"""

import sys
import argparse
import os.path
from pickle import load
sys.path.insert(0, '..')
from analyze_visual.analyze_visual import process_video


def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Wrapper"
                                                 "Predict shot class")

    parser.add_argument("-i", "--input_videos_path",
                        required=True, nargs=None,
                        help="Videos folder path")

    parser.add_argument("-m", "--model", required=True, nargs=None,
                        help="Model")

    return parser.parse_args()


def video_class_predict(features, algorithm):
    """
    Loads pre-trained model and predict single shot's class
    :param features: features
    :algorithm: Training algorithm
    :return:    
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


def main(argv):

    args = parse_arguments()
    videos_path = args.input_videos_path
    algorithm = args.model
    final_proba = np.empty((0, 2))
    if os.path.isfile(videos_path):
        features_stats = process_video(videos_path, 2, True, True, True)
        features = features_stats[0]
        features = features.reshape(1, -1)
        # Predict the classes of shots
        probas, classes = video_class_predict(features, algorithm)
        for class_name, proba in zip(classes, probas):
            print(f'Video {videos_path} belongs by {proba} in {class_name} class')
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
            probas, classes = video_class_predict(features, algorithm)
            #Save the resuls in a numpy array
            final_proba = np.append(final_proba,[probas],axis=0)
            #Save results of every video in a text file
            with open(str(videos_path)+".txt", "a") as text_file:
                for class_name, proba in zip(classes, probas):
                    print(f'Video {v} belongs by {proba} in {class_name} class', file=text_file)  
        final_proba=final_proba.mean(axis=0)
        #Print and save the final results
        with open(str(videos_path)+".txt", "a") as text_file:
            for class_name, proba in zip(classes, final_proba):
                print(f'The movie {videos_path} belongs by {"{:.2%}".format(proba)} in {class_name} class', file=text_file)
                print(f'The movie {videos_path} belongs by {"{:.2%}".format(proba)} in {class_name} class')   


if __name__ == '__main__':
    main(sys.argv)
