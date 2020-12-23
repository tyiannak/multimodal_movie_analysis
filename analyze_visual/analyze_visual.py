"""
Visual Analysis.

The script extracts visual-based features from videos.

Flags:

    - f: extract features from specific file
        E.g. python3 analyze_visual.py -f <filename>
    - d: extract features from all files of a directory
        E.g. python3 analyze_visual.py -d <directory_name>

Basic functions:

    - process_video :  extracts features from specific file
    - dir_process_video : extracts features from all files of a directory
    - dirs_process_video : extracts features from all files of different directories

Please read the docstrings for further information.
"""

import cv2
import time
import sys
import glob
import os
import numpy as np
import collections

from object_detection import detection_utils as dutils
from object_detection import generic_model as gmodel
from utils import *


generic_model = gmodel.SsdNvidia()


def process_video(video_path, process_mode, print_flag=True,
                  online_display=False,  save_results=True):
    """
    Extracts and displays features representing color, flow, objects detected
    and shot duration from video

    Args:

        video_path (str) : Path to video file
        process_mode (int) : Processing modes:
            - 0 : No processing
            - 1 : Color analysis
            - 2 : Flow analysis and object detection
        print_flag (bool) : Flag to allow the display of terminal messages.
        online_display (bool): Flag to allow the display of online video features
        save_results (bool) : Boolean variable to allow save results files.

    Returns:

        features_stats (array_like) : Feature vector with stats on features
            over time. Stats:
                - mean value of every feature over time
                - standard deviation of every feature over time
                - mean value of standard deviation of every feature over time
                - mean value of the 10 highest-valued frames for every feature
        feature matrix (array_like) : Array of the extracted features.
            Contains one feature vector for every frame.
            If object detection is used, the feature matrix contains information
            for the first n = (number_of_frames - max_frames + 1) frames.
        feature_names (tuple) : 2D tuple that contains the name
            of the extracted features.
            - feature_names[0] (list) : names of the feature_matrix features
            - feature_names[1] (list) : names of the feature_stats features
    """

    # ---Initializations-------------------------------------------------------
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    t_start = time.time()
    t_0 = t_start
    capture = cv2.VideoCapture(video_path)
    frames_number = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = capture.get(cv2.CAP_PROP_FPS)
    duration_secs = frames_number / fps
    hrs, mins, secs, tenths = seconds_to_time(duration_secs)
    duration_str = '{0:02d}:{1:02d}:{2:02d}.{3:02d}'.format(hrs, mins,
                                                            secs, tenths)
    if print_flag:
        print('Began processing video : ' + video_path)
        print("FPS      = " + str(fps))
        print("Duration = " + str(duration_secs) + " - " + duration_str)

    p_old = np.array([])
    time_stamps = np.array([])
    f_diff = np.array([])
    fps_process = np.array([])
    t_process = np.array([])

    if process_mode > 1:
        # --------------------------------------------------------------------
        which_object_categories = 2  # which object categories to store as features
        """
        Takes values:
            0: returns features for all 80 categories
            1: returns features for 12 super categories
            2: returns features for both 80 and 12 categories

        """
        # --------------------------------------------------------------------
        overlap_threshold = 0.8
        mean_confidence_threshold = 0.5
        max_frames = 3
        objects_boxes_all = []
        objects_labels_all = []
        objects_confidences_all = []
        frontal_faces_num = collections.deque(maxlen=200)
        frontal_faces_ratio = collections.deque(maxlen=200)
        tilt_pan_confidences = collections.deque(maxlen=200)

        cascade_frontal, cascade_profile = initialize_face(
            HAAR_CASCADE_PATH_FRONTAL, HAAR_CASCADE_PATH_PROFILE)
    count = 0
    count_process = 0

    next_process_stamp = 0.0
    process_now = False
    shot_change_times = [0]
    shot_change_process_indices = [0]
    shot_durations = []

    # ---Calculate features for every frame-----------------------------------
    while capture.isOpened():
        # cv.SetCaptureProperty( capture, cv.CV_CAP_PROP_POS_FRAMES,
        # count*frameStep );
        # (THIS IS TOOOOO SLOW (MAKES THE READING PROCESS 2xSLOWER))

        # get frame
        ret, frame = capture.read()
        time_stamp = float(count) / fps
        if time_stamp >= next_process_stamp:
            next_process_stamp += process_step
            process_now = True

        # ---Begin processing-------------------------------------------------
        if ret:
            count += 1
            frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = resize_frame(frame2, new_width)
            img_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            (width, height) = img_gray.shape[1], img_gray.shape[0]

            if process_mode > 1:
                if (count % 25) == 1:
                    # Determines strong corners on an image.
                    p0 = cv2.goodFeaturesToTrack(img_gray, mask=None,
                                                 **feature_params)
                    if p0 is None:
                        p0 = p_old
                    p_old = p0

            if process_now:

                count_process += 1
                time_stamps = np.append(time_stamps, time_stamp)
                feature_vector_current = np.array([])

                # ---Get features from color analysis-------------------------
                if process_mode > 0:
                    # PROCESS LEVEL 1:
                    feature_vector_current,\
                     hist_rgb_ratio, hist_s, hist_v,\
                     v_norm, _ = color_analysis(feature_vector_current, rgb)

                    if count_process > 1:
                        f_diff = np.append(f_diff,
                                           np.mean(
                                               np.mean(
                                                   np.abs(
                                                       hist_v - hist_v_prev))))
                    else:
                        f_diff = np.append(f_diff, 0.0)

                    feature_vector_current = np.concatenate(
                        (feature_vector_current,
                         np.array([f_diff[-1]])),
                        0)
                    hist_v_prev = hist_v

                # ---Get flow and object related features---------------------
                if process_mode > 1:
                    # face detection
                    frontal_faces = detect_faces(rgb, cascade_frontal,
                                                 cascade_profile)
                    # update number of faces
                    frontal_faces_num, frontal_faces_ratio = update_faces(
                        width * height, frontal_faces,
                        frontal_faces_num, frontal_faces_ratio)

                    # ---Get tilt/pan confidences-----------------------------
                    if count_process > 1 and len(p0) > 0:
                        angles, mags, \
                         mu, std, good_new,\
                         good_old, dx_all, dy_all, \
                         tilt_pan_confidence = flow_features(
                                    img_gray, img_gray_prev, p0, lk_params)
                        mag_mu = np.mean(np.array(mags))
                        mag_std = np.std(np.array(mags))
                    else:
                        tilt_pan_confidence = 0.0
                        mag_mu = 0
                        mag_std = 0
                    tilt_pan_confidences.append(tilt_pan_confidence)

                    # ---Get shot duration------------------------------------
                    if count_process > 1:
                        gray_diff = (img_gray_prev - img_gray)
                        if shot_change(gray_diff, mag_mu, f_diff,
                                       time_stamp - shot_change_times[-1]):
                            shot_change_times.append(time_stamp)
                            shot_change_process_indices.append(count_process)

                            shot_durations = calc_shot_duration(
                                shot_change_times,
                                shot_change_process_indices,
                                shot_durations)

                    # add new features to feature vector
                    feature_vector_current = np.concatenate(
                        (feature_vector_current,
                         np.array([frontal_faces_num[-1]]),
                         np.array([frontal_faces_ratio[-1]]),
                         np.array([tilt_pan_confidences[-1]]),
                         np.array([mag_mu]),
                         np.array([mag_std])),
                        0)

                # ---Append current feature vector to feature matrix----------
                if process_mode > 0:
                    if count_process == 1:
                        feature_matrix = np.reshape(
                                    feature_vector_current,
                                    (1, len(feature_vector_current)))
                    else:
                        feature_matrix = np.concatenate(
                            (feature_matrix,
                             np.reshape(
                                 feature_vector_current,
                                 (1,
                                  len(feature_vector_current)))),
                            0)

                # ---Display features on windows------------------------------

                if online_display and (count_process > 2) and\
                        (count_process % plot_step == 0) and print_flag:
                    # draw RGB image and visualizations
                    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    if process_mode > 1 and len(p0) > 0:
                        # faces bounding boxes:
                        for f in frontal_faces:  # draw face rectangles
                            cv2.rectangle(vis, (f[0], f[1]),
                                          (f[0] + f[2], f[1] + f[3]),
                                          (0, 255, 255), 3)

                        # draw motion arrows
                        for i, (new, old) in enumerate(
                                zip(good_new, good_old)):
                            vis = cv2.arrowedLine(
                                vis, tuple(new), tuple(old),
                                color=(0, 255, 0), thickness=1)

                        if len(angles) > 0:
                            vis = cv2.arrowedLine(
                                vis, (int(width / 2), int(height / 2)),
                                (int(width / 2) + int(np.mean(dx_all)),
                                 int(height / 2) + int(np.mean(dy_all))),
                                color=(0, 0, 255), thickness=4, line_type=8,
                                shift=0)
                        cv2.putText(vis, str(int(mu)), (int(width / 2),
                                                        int(height / 2)),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, 255)

                    # Time-related plots:
                    dur_secs = float(count) / fps
                    t_2 = display_time(dur_secs, fps_process, t_process,
                                       t_0, duration_secs, duration_str,
                                       width, vis)

                    # Display features on windows
                    windows_display(vis, height, process_mode, v_norm,
                                    hist_rgb_ratio, hist_v, hist_s,
                                    frontal_faces_num, frontal_faces_ratio,
                                    tilt_pan_confidences)

                    window_name = 'Object Detection'
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, (width, height))

                    objects = generic_model.detect(frame, 0.4)

                    generic_model.display_cv2(frame, objects, window_name)
                    cv2.moveWindow(window_name, width, 0)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                    t_0 = t_2

                objects = generic_model.detect(frame, 0.1)

                objects_boxes_all.append(objects[0])
                objects_labels_all.append(objects[1])
                objects_confidences_all.append(objects[2])

                process_now = False
                img_gray_prev = img_gray

        else:
            capture.release()
            cv2.destroyAllWindows()

    processing_time = time.time() - t_start

    # ---Append shot durations in feature matrix------------------------------
    for ccc in range(count_process - shot_change_process_indices[-1]):
        shot_durations.append(count_process - shot_change_process_indices[-1])

    shot_durations = np.matrix(shot_durations)
    shot_durations = shot_durations * process_step
    feature_matrix = np.append(feature_matrix, shot_durations.T, axis=1)

    # Get movie-level feature statistics
    # TODO: consider more statistics, OR consider temporal analysis method
    # eg LSTMs or whatever
    features_stats = get_features_stats(feature_matrix)

    # ---Print and save the outcomes------------------------------------------
    if print_flag:
        processing_fps = count_process / float(processing_time)
        processing_rt = 100.0 * float(processing_time) / duration_secs
        hrs, mins, secs, tenths = seconds_to_time(processing_time)

        print('Finished processing on video :' + video_path)
        print("processing time: " + '{0:02d}:{1:02d}:{2:02d}.{3:02d}'.
              format(hrs, mins, secs, tenths))
        print("processing ratio      {0:3.1f} fps".format(processing_fps))
        print("processing ratio time {0:3.0f} %".format(processing_rt))
        print('Actual shape of feature matrix without object features: {}'.format(
            feature_matrix.shape))
        print('Shape of features\' stats found: {}'.format(
            features_stats.shape))
        print('Number of shot changes: {}'.format(len(shot_change_times)))
    if process_mode > 1:
        objects = dutils.smooth_object_confidence(
                    objects_labels_all, objects_confidences_all,
                    objects_boxes_all, overlap_threshold,
                    mean_confidence_threshold, max_frames)

        if objects:
            out_labels, out_boxes, out_confidences = objects
            (object_features_stats,
             super_object_features_stats) = dutils.get_object_features(
                out_labels, out_confidences, out_boxes, which_object_categories)

            #dutils.save_object_features(object_features_stats, super_object_features_stats,
            #                     2)

            (labels_freq_per_frame,
             labels_avg_confidence_per_frame, labels_area_ratio_per_frame) = \
                dutils.get_object_features_per_frame(
                    out_labels, out_confidences, out_boxes,
                    which_object_categories)
            feature_matrix = np.concatenate((
                feature_matrix[:(-max_frames + 1)][:],
                labels_freq_per_frame), axis=1)
            feature_matrix = np.concatenate((
                feature_matrix, labels_avg_confidence_per_frame), axis=1)
            feature_matrix = np.concatenate((
                feature_matrix, labels_area_ratio_per_frame), axis=1)

            if which_object_categories > 0:
                overall_labels_freq = np.asarray(super_object_features_stats[0])
                overall_labels_avg_confidence = np.asarray(super_object_features_stats[1])
                overall_labels_area_ratio = np.asarray(super_object_features_stats[2])
                features_stats = np.concatenate((
                    features_stats, overall_labels_freq,
                    overall_labels_avg_confidence, overall_labels_area_ratio))

            else:
                overall_labels_freq = np.asarray(object_features_stats[0])
                overall_labels_avg_confidence = np.asarray(object_features_stats[1])
                overall_labels_area_ratio = np.asarray(object_features_stats[2])
                features_stats = np.concatenate((
                    features_stats, overall_labels_freq,
                    overall_labels_avg_confidence, overall_labels_area_ratio))

            if print_flag:
                print('Shape of feature matrix including '
                      'object features (after smoothing'
                      ' object confidences): {}'.format(feature_matrix.shape))
                print('Shape of feature stats vector including'
                      ' object features (after smoothing'
                      ' object confidences): {}'.format(features_stats.shape))

        if save_results:
            np.savetxt("feature_matrix.csv", feature_matrix, delimiter=",")
            np.savetxt("features_stats.csv", features_stats, delimiter=",")

    f_names, f_names_stats = get_features_names(process_mode,
                                                which_object_categories)

    return features_stats,  f_names_stats,  feature_matrix, f_names, \
           shot_change_times


def dir_process_video(dir_name, process_mode, print_flag,
                      online_display,  save_results):
    """Processes all videos from a given directory."""
    
    dir_name_no_path = os.path.basename(os.path.normpath(dir_name))

    features_all = []

    types = ('*.avi', '*.mpeg', '*.mpg', '*.mp4', '*.mkv', '*.webm')
    video_files_list = []
    for files in types:
        video_files_list.extend(glob.glob(os.path.join(dir_name, files)))
    video_files_list = sorted(video_files_list)

    for movieFile in video_files_list:
        print(movieFile)
        features_stats, f_names_stats, feature_matrix, f_names, shot_change_t = \
            process_video( movieFile, process_mode, print_flag, online_display,
                           save_results)
        np.save(movieFile + ".npy", feature_matrix)
        print('vehicle_freq')
        print(features_stats[f_names_stats.index('vehicle_freq')])
        # 'vehicle_mean_confidence', 'vehicle_mean_area_ratio'

        features_all.append(features_stats)

    np.save(os.path.join(dir_name,dir_name_no_path + "_features.npy"), features_all)
    np.save(os.path.join(dir_name,dir_name_no_path + "_video_files_list.npy"), video_files_list)
    features_all = np.array(features_all)

    return features_all, video_files_list
    
  
def main(argv):

    process_mode = 2
    print_flag = True
    online_display = True
    save_results = True
    if len(argv) == 3:
        if argv[1] == "-f":
            f_stat, f_stat_n, f, f_n, shots = process_video(argv[2],
                                                            process_mode,
                                                            print_flag,
                                                            online_display,
                                                            save_results)
        elif argv[1] == "-d":  # directory
            dir_name = argv[2]
            features_all, video_files_list = dir_process_video(dir_name,
                                                               process_mode,
                                                               print_flag,
                                                               online_display,
                                                               save_results)
            print(features_all.shape)
            print(video_files_list)
        else:
            print('Error: Unsupported flag.')
            print('For supported flags please read the modules\' docstring.')
    elif len(argv) < 3:
        print('Error: There are 3 required arguments, but less were given.')
        print('For help please read the modules\' docstring.')
    else:
        print('Error: There are 3 required arguments, but more were given.')
        print('For help please read the modules\' docstring.')


if __name__ == '__main__':
    main(sys.argv)
