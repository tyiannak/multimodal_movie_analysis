import sys
import pickle
import argparse
import numpy as np
from train import data_preparation
sys.path.insert(0, '..')
from analyze_visual.analyze_visual import process_video, dir_process_video

def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Create Shot "
                                                 "Classification Dataset")
    parser.add_argument("-f", "--file", required=True, nargs=None,
                        help="File")

    return parser.parse_args()

def model_predict(features):

    loaded_model = pickle.load(open('trained_svm.sav', 'rb'))

    result = loaded_model.predict(features)

    print(result)


if __name__ == "__main__":

    args = parse_arguments()
    shot = args.file

    f_stat, f_stat_n, feature_matrix, f_n, shots = process_video(shot,
                                                                 2, True, True, True)
    #feature_matrix = np.load('dataset/Tilt/Absolute Power cd1.avi_shot_3175_3178..avi.mp4.npy')

    model_predict(f_stat)



    

    