"""
This script is used to train audio segment classifiers used in movie analysis

Usage example:

TODO

"""

import argparse
import sys
from pyAudioAnalysis import audioTrainTest as aT
sys.path.insert(0, '..')

def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Train audio classifiers")
    parser.add_argument("-i", "--input_audio", required=True,
                        nargs='+', help="List of audio paths")
    parser.add_argument("-o", "--output_model", required=True,
                        nargs=None, help="Output model's path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    input_audio = args.input_audio
    output_model = args.output_model

    mt_win = 3.0
    mt_step = 1.0
    # this is obviously not optimal in terms of performance but it is quite fast
    st_win = st_step = 0.1

    aT.extract_features_and_train(input_audio, mt_win, mt_step,
                                  st_win, st_step,
                                  "svm_rbf", output_model, False)

