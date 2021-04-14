"""
Audio Analysis
The script extracts audio-based features from movies.
Flags:
    - f: extract features from specific file
        E.g. python3 analyze_audio.py -f <filename>
    - d: extract features from all files of a directory
        E.g. python3 analyze_audio.py -d <directory_name>
Basic functions:
    - process_audio:  extracts features from specific file
    - dir_process_audio : extracts features from all files of a directory
    - dirs_process_audio : extracts features from all files of different directories
Please read the docstrings for further information.
"""

import glob
import os
import numpy as np
import sys
import os.path
from pyAudioAnalysis.audioSegmentation import mid_term_file_classification \
    as mtf

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))



def process_audio(audio_path, save_results=True):
    """

    """
    models_path = "segment_models"
    flag_generic, class_generic, _, _ = mtf(audio_path,
                                            os.path.join(models_path,
                                                         "audio_4class"),
                                            "svm_rbf", False)

    flag_speech_ar, class_speech_ar, _, _ = mtf(audio_path,
                                                os.path.join(models_path,
                                                             "speech_arousal"),
                                                "svm_rbf", False)

    flag_speech_val, class_speech_val, _, _ = mtf(audio_path,
                                                   os.path.join(models_path,
                                                                "speech_valence"),
                                                  "svm_rbf", False)

    flag_mus_en, class_mus_en, _, _ = mtf(audio_path,
                                          os.path.join(models_path,
                                                       "music_energy"),
                                          "svm_rbf", False)

    flag_mus_val, class_mus_val, _, _ = mtf(audio_path,
                                            os.path.join(models_path,
                                                         "music_valence"),
                                            "svm_rbf", False)

    features = {}

    for ic, c in enumerate(class_generic):
        features[c] = np.count_nonzero(flag_generic == float(ic))
        features[c] /= len(flag_generic)

    for ic, c in enumerate(class_speech_ar):
        features["speech_arousal_" + c] = np.count_nonzero(
            (flag_speech_ar == float(ic)) &
            (flag_generic == float(class_generic.index("speech"))))
        features["speech_arousal_" + c] /= (np.count_nonzero(flag_generic ==
                                                             float(class_generic.index("speech")))
                                            + 0.001)

    for ic, c in enumerate(class_speech_val):
        features["speech_valence_" + c] = np.count_nonzero(
            (flag_speech_val == float(ic)) &
            (flag_generic == float(class_generic.index("speech"))))
        features["speech_valence_" + c] /= (np.count_nonzero(flag_generic ==
                                                             float(class_generic.index("speech")))
                                            + 0.001)

    for ic, c in enumerate(class_mus_en):
        features["music_energy_" + c] = np.count_nonzero(
            (flag_mus_en == float(ic)) &
            (flag_generic == float(class_generic.index("music"))))
        features["music_energy_" + c] /= (np.count_nonzero(flag_generic ==
                                                             float(class_generic.index("music")))
                                            + 0.001)

    for ic, c in enumerate(class_mus_val):
        features["music_valence_" + c] = np.count_nonzero(
            (flag_mus_val == float(ic)) &
            (flag_generic == float(class_generic.index("music"))))
        features["music_valence_" + c] /= (np.count_nonzero(flag_generic ==
                                                             float(class_generic.index("music")))
                                            + 0.001)

    return features.values(), features.keys()


def dir_process_audio(dir_name):
    dir_name_no_path = os.path.basename(os.path.normpath(dir_name))

    features_all = []

    types = ('*.wav', )
    video_files_list = []
    for files in types:
        video_files_list.extend(glob.glob(os.path.join(dir_name, files)))
    video_files_list = sorted(video_files_list)

    for movieFile in video_files_list:
        print(movieFile)
        features_stats, f_names_stats = process_audio(movieFile,)
        features_all.append(features_stats)

    np.save(os.path.join(dir_name,dir_name_no_path + "_features.npy"),
            features_all)
    np.save(os.path.join(dir_name,dir_name_no_path + "_video_files_list.npy"),
            video_files_list)
    np.save(os.path.join(dir_name,dir_name_no_path + "_f_names.npy"),
            f_names_stats)
    features_all = np.array(features_all)

    return features_all, video_files_list, f_names_stats


def main(argv):
    if len(argv) == 3:
        if argv[1] == "-f":
            f_stat, f_stat_n = process_audio(argv[2])
            print(f_stat, f_stat_n)

        elif argv[1] == "-d":  # directory
            dir_name = argv[2]
            features_all, video_files_list, f_names = \
                dir_process_audio(dir_name)
            print(features_all.shape)
            print(video_files_list)
        else:
            print('Error: Unsupported flag.')
            print('For supported flags please read the modules\' docstring.')


if __name__ == '__main__':
    main(sys.argv)