"""
Used to evaluate shot video detection.

File-level example:


Dir-level example: (
ground truth is supposed to be under the same folder with same filename as the
video file with ".txt" appended at the end)
python3 shot_evaluation.py -d data
"""

from analyze_visual import process_video
import numpy as np
import os
import sys
import glob


def read_gt_file(file):
    """
    Read the shot detection ground-truth file
    The file has the format:
    0
    <shot_change_1_timestamp_in_seconds>
    <shot_change_2_timestamp_in_seconds>
    ...
    """
    with open(file) as w:
        lines = w.read().splitlines()
    for i in range(0, len(lines)):
        lines[i] = float(lines[i])
    return lines


def calc(annotated_shots,shot_change_times):
    tolerance, correct_shot, correct_shot_h = 0.5, 0, 0

    correct_recall, correct_precision = 0, 0
    for t in annotated_shots:
        if np.abs(t - np.array(shot_change_times)).min() < tolerance:
            correct_recall += 1

    for t in shot_change_times:
        if np.abs(t - np.array(annotated_shots)).min() < tolerance:
            correct_precision += 1

    precision= correct_precision/(len(shot_change_times))
    recall = correct_recall/(len(annotated_shots))

    print("Precision: {0:.0f}%".format(precision*100))
    print("Recall: {0:.0f}%".format(recall*100))

    return precision, recall


def single_acc(video_path, shot_change_times):
    annotated_shots = read_gt_file(video_path + ".txt")
    for i in range(0, len(shot_change_times)): 
        shot_change_times[i] = float(shot_change_times[i])
    
    print("Timestamps for predicted shots: \n", shot_change_times)
    print("Timestamps for actual shots: \n", annotated_shots)
    calc(annotated_shots, shot_change_times)


def dir_acc(video_path):
    types = ('*.avi', '*.mpeg', '*.mpg', '*.mp4', '*.mkv')
    video_files_list = []
    prec_list, rec_list=[], []

    for files in types:
        video_files_list.extend(glob.glob(os.path.join(video_path, files)))
    video_files_list = sorted(video_files_list)
    print(video_files_list)

    for movie_file in video_files_list:
        shot_file = movie_file + ".txt"
        print(shot_file)
        if os.path.isfile(shot_file):
            features_stats, f_names_stats, feature_matrix, f_names, \
            shot_change_t = process_video(movie_file, 2, True, False)
            annotated_shots = read_gt_file(shot_file)
            print("Timestamps for predicted shots: \n", shot_change_t)
            print("Timestamps for actual shots: \n", annotated_shots)
            precision, recall = calc(annotated_shots, shot_change_t)
            prec_list.append(precision)
            rec_list.append(recall)
            shot_change_t.clear()
    print("AVG of Precision: {0:.0f}%".format(np.mean(prec_list)*100))
    print("AVG of Recall: {0:.0f}%".format(np.mean(rec_list)*100))


def main(argv):
    if len(argv) == 3:
        if argv[1] == "-f":
            _, _, shot_change_t = process_video(argv[2], 2, True, True)
            single_acc(argv[2], shot_change_t)
        elif argv[1] == "-d":  # directory
            dir_acc(argv[2])
        else:
            print("Wrong syntax. Check docstrings")
    else:
        print("Wrong syntax. Check docstrings")


if __name__ == '__main__':
    main(sys.argv)
