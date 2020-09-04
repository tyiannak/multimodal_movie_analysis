import cv2
import time
import sys
import glob
import os
import numpy as np
import scipy.spatial.distance as dist
import collections

# process and plot related parameters:
new_width = 500
process_step = 0.5
plot_step = 2

# face detection-related paths:
HAAR_CASCADE_PATH_FRONTAL = "haarcascade_frontalface_default.xml"
HAAR_CASCADE_PATH_PROFILE = "haarcascade_frontalface_default.xml"

# flow-related parameters:
lk_params = dict(winSize=(15, 15),
                 maxLevel=5,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                           10, 0.03))
feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=3,
                      blockSize=5)


def angle_diff(angle1, angle2):
    """Returns difference between 2 angles."""

    diff = np.abs(angle2 - angle1)
    if np.abs(diff) > 180:
        diff -= 360
    return diff


def angles_std(angles: np.ndarray or list, mu: float) -> float:
    """
    Args:
        angles (np.ndarray or list): list of angles
        mu (float): mean value of the angles

    Returns the standard deviation (std) of a set of angles.
    """

    std = 0.0
    for a in angles:
        std += (angle_diff(a, mu) ** 2)
    std /= len(angles)
    std = np.sqrt(std)
    return std


def display_histogram(data, width, height, maximum, window_name):
    """
    Calculates and dispalys on a window
    the histogram of the data, under
    specific maximum value

    Args:
        data (ndarray): input data
        width (int): width of the displayed window
        height (int): width of the displayed window
        maximum : maximum number on data
        window_name (str) : name of the displayed window

    Returns:
        None
    """

    if len(data) > width:
        hist_item = height * (data[len(data) - width - 1:-1] / maximum)
    else:
        hist_item = height * (data / maximum)
    img = np.zeros((height, width, 3))
    hist = np.int32(np.around(hist_item))

    for x, y in enumerate(hist):
        cv2.line(img, (x, height), (x, height - y), (255, 0, 255))

    cv2.imshow(window_name, img)
    return


def intersect_rectangles(r1, r2):
    """
    Args:
        r1: 4 coordinates of the first rectangle
        r2: 4 coordinates of the second rectangle

    Returns:
        e_ratio: ratio of intersectance
    """

    x1 = max(r1[0], r2[0])
    x2 = min(r1[0] + r1[2], r2[0] + r2[2])
    y1 = max(r1[1], r2[1])
    y2 = min(r1[1] + r1[3], r2[1] + r2[3])

    w = x2 - x1
    h = y2 - y1
    if (w > 0) and (h > 0):
        e = w * h
    else:
        e = 0.0
    e_ratio = 2.0 * e / (r1[2] * r1[3] + r2[2] * r2[3])
    return e_ratio


def initialize_face(frontal_path, profile_path):
    """Reads and returns frontal and profile
    haarcascade classifiers from paths."""

    cascade_frontal = cv2.CascadeClassifier(frontal_path)
    cascade_profile = cv2.CascadeClassifier(profile_path)
    return cascade_frontal, cascade_profile


def remove_overlaps(rectangles):
    """
    Removes overlaped rectangles

    Args:
        rectangles (list) : list of lists containing rectangles coordinates

    Returns:
        List of non overlapping rectangles.
    """

    for i, rect_i in enumerate(rectangles):
        for j, rect_j in enumerate(rectangles):
            if i != j:
                inter_ratio = intersect_rectangles(rect_i,
                                                   rect_j)
                if inter_ratio > 0.3:
                    del rectangles[i]
                    break

    return rectangles


def detect_faces(image, cascade_frontal, cascade_profile):
    """
    Detects faces on image. Temporarily only detects frontal face.

    Args:
        image : image of interest
        cascade_frontal : haar cascade classifier for fronatl face
        cascade_profile : haar cascade classifier for profile face

    Returns:
        faces_frontal (list) : list of rectangles coordinates
    """

    faces_frontal = []
    detected_frontal = cascade_frontal.detectMultiScale(image, 1.3, 5)
    print(detected_frontal)
    if len(detected_frontal) > 0:
        for (x, y, w, h) in detected_frontal:
            faces_frontal.append((x, y, w, h))

    faces_frontal = remove_overlaps(faces_frontal)
    return faces_frontal


def resize_frame(frame, target_width):
    """
    Resizes a frame according to specific width.

    Args:
        frame : frame to resize
        target_width : width of the final frame

    """
    width, height = frame.shape[1], frame.shape[0]

    if target_width > 0:  # Use Framewidth = 0 for NO frame resizing
        ratio = float(width) / target_width
        new_height = int(round(float(height) / ratio))
        frame_final = cv2.resize(frame, (target_width, new_height))
    else:
        frame_final = frame

    return frame_final


def get_rgb_histograms(image_rgb):
    """Computes Red, Green and Blue histograms of an RGB image."""

    # compute histograms:
    [hist_r, _] = np.histogram(image_rgb[:, :, 0], bins=range(-1, 256, 32))
    [hist_g, _] = np.histogram(image_rgb[:, :, 1], bins=range(-1, 256, 32))
    [hist_b, _] = np.histogram(image_rgb[:, :, 2], bins=range(-1, 256, 32))

    # normalize histograms:
    hist_r = hist_r.astype(float) / np.sum(hist_r.astype(float))
    hist_g = hist_g.astype(float) / np.sum(hist_g.astype(float))
    hist_b = hist_b.astype(float) / np.sum(hist_b.astype(float))

    return hist_r, hist_g, hist_b


def get_hsv_histograms(image_hsv):
    """Computes Hue, Saturation and Value histograms of an HSV image."""

    # compute histograms:
    [hist_h, _] = np.histogram(image_hsv[:, :, 0], bins=range(180))
    [hist_s, _] = np.histogram(image_hsv[:, :, 1], bins=range(256))
    [hist_v, _] = np.histogram(image_hsv[:, :, 2], bins=range(256))

    # normalize histograms:
    hist_h = hist_h.astype(float)
    hist_h = hist_h / np.sum(hist_h)
    hist_s = hist_s.astype(float)
    hist_s = hist_s / np.sum(hist_s)
    hist_v = hist_v.astype(float)
    hist_v = hist_v / np.sum(hist_v)

    return hist_h, hist_s, hist_v


def get_hsv_histograms_2d(image_hsv):
    width, height = image_hsv.shape[1], image_hsv.shape[0]
    h, x_edges, y_edges = np.histogram2d(np.reshape(image_hsv[:, :, 0],
                                                    width * height),
                                         np.reshape(image_hsv[:, :, 1],
                                                    width * height),
                                         bins=(range(-1, 180, 30),
                                               range(-1, 256, 64)))
    h = h / np.sum(h)
    return h, x_edges, y_edges


def flow_features(img_gray, img_gray_prev, p0, params):
    """
    Calculates the flow of specific points between two images

    Args:
        img_gray : current image on gray scale
        img_gray_prev : previous image on gray scale
        p0 : vector of 2D points for which the flow needs to be found;
            point coordinates must be single-precision floating-point numbers
        params : parameters dictionary for cv2.calcOpticalFlowPyrLK function

    Returns:
        angles : ndarray of angles for the flow vectors
        mags : ndarray -?-
        mu : mean value of the angles
        std : standard deviation of the angles
        good_new : a 2D vector of the new position of the input points
        good_old : the input vector of 2D points
        dx_all : list of differences between old and new points for the x axis
        dy_all : list of differences between old and new points for the y axis
        tilt_pan_confidence : tilt/pan confidence of the camera

    """
    # get new position of the input points
    p1, st, err = cv2.calcOpticalFlowPyrLK(img_gray_prev, img_gray, p0,
                                           None, **params)
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    angles = []
    mags = []
    dx_all = []
    dy_all = []

    # find angles, mags and distances
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        x1, y1 = new.ravel()
        x2, y2 = old.ravel()
        dx = x2 - x1
        dy = y2 - y1

        if dy < 0:
            angles.append([np.abs(180.0 * np.arctan2(dy, dx) / np.pi)])
        else:
            angles.append([360.0 - 180.0 * np.arctan2(dy, dx) / np.pi])

        mags.append(np.sqrt(dx ** 2 + dy ** 2) /
                    np.sqrt(img_gray.shape[0] ** 2 +
                            img_gray.shape[1] ** 2))
        dx_all.append(dx)
        dy_all.append(dy)

    angles = np.array(angles)
    mags = np.array(mags)
    dist_horizontal = -1

    # find mu, std and tilt_pan_confidence
    if len(angles) > 0:
        mean_dx = np.mean(dx_all)
        mean_dy = np.mean(dy_all)
        if mean_dy < 0:
            mu = -(180.0 * np.arctan2(int(mean_dy),
                                      int(mean_dx)) / np.pi)
        else:
            mu = 360.0 - (180.0 * np.arctan2(int(mean_dy),
                                             int(mean_dx)) / np.pi)
        std = angles_std(angles, mu)

        dist_horizontal = min(angle_diff(mu, 180), angle_diff(mu, 0))
        tilt_pan_confidence = np.mean(mags) / np.sqrt(std + 0.00000001)
        tilt_pan_confidence = tilt_pan_confidence[0]
        # TODO:
        # CHECK PANCONFIDENCE
        # SAME FOR ZOOM AND OTHER CAMERA EFFECTS

        if tilt_pan_confidence < 1.0:
            tilt_pan_confidence = 0
            dist_horizontal = -1
    else:
        mags = [0]
        angles = [0]
        dx_all = [0]
        dy_all = [0]
        mu = 0
        std = 0
        tilt_pan_confidence = 0.0

    return angles, mags, mu, std, good_new, \
        good_old, dx_all, dy_all, tilt_pan_confidence


def seconds_to_time(duration_secs):
    """Converts seconds to hours : minutes : seconds : tenths of seconds"""

    hrs = int(duration_secs / 3600)
    mins = int(duration_secs / 60)
    secs = int(duration_secs) % 60
    tenths = int(100 * (duration_secs - int(duration_secs)))

    return hrs, mins, secs, tenths


def color_analysis(feature_vector_old, rgb):
    """
    Extract color related features from rgb image.
    Features to be extracted:
        - From RGB image:
            - Histogram of red
            - Histogram of green
            - Histogram of blue
            - Histogram of ratio of the max difference
                from the mean value of the image
        - From HSV image:
            - Histogram of hue
            - Histogram of saturation
            - Histogram of value

    Args:
        feature_vector_old (array_like) : Old feature vector
                                            to be concatenated
        rgb : Image of interest in RGB format.

    Returns:
        feature_vector_new : Concatenated feature vector
        hist_rgb_ratio : Histogram of rgb ratio
        hist_s : Histogram of saturation
        hist_v : Histogram of value
        v_norm : Value norm
        s_norm : Saturation norm
    """

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    [hist_r, hist_g, hist_b] = get_rgb_histograms(rgb)

    rgb_ratio = 100.0 * (np.max(rgb, 2) -
                         np.mean(rgb, 2)) / (1.0 + np.mean(rgb, 2))
    v_norm = (255.0 * hsv[:, :, 2]) / np.max(hsv[:, :, 2] + 1.0)
    s_norm = (255.0 * hsv[:, :, 1]) / np.max(hsv[:, :, 1] + 1.0)

    rgb_ratio[rgb_ratio > 199.0] = 199.0
    rgb_ratio[rgb_ratio < 1.0] = 1.0
    hist_rgb_ratio, _ = np.histogram(rgb_ratio,
                                     bins=range(-1, 200, 40))
    hist_rgb_ratio = hist_rgb_ratio.astype(float)
    hist_rgb_ratio = hist_rgb_ratio / np.sum(hist_rgb_ratio)
    hist_v, _ = np.histogram(v_norm, bins=range(-1, 256, 32))
    hist_v = hist_v.astype(float)
    hist_v = hist_v / np.sum(hist_v)
    hist_s, _ = np.histogram(s_norm, bins=range(-1, 256, 32))
    hist_s = hist_s.astype(float)
    hist_s = hist_s / np.sum(hist_s)

    # update the current feature vector
    feature_vector_new = np.concatenate(
        (feature_vector_old, hist_r, hist_g,
         hist_b, hist_v, hist_rgb_ratio, hist_s),
        0)

    return feature_vector_new, hist_rgb_ratio, hist_s, hist_v, v_norm, s_norm


def update_faces(area, frontal_faces, frontal_faces_num, frontal_faces_ratio):
    """
    Updates frontal faces number and frontal faces ratio
    for the new frame.
    """

    frontal_faces_num.append(float(len(frontal_faces)))
    if len(frontal_faces) > 0:
        f_tmp = 0.0
        for f in frontal_faces:
            # normalize face size ratio to the frame dimensions
            f_tmp += (f[2] * f[3] / float(area))
        frontal_faces_ratio.append(f_tmp / len(frontal_faces))
    else:
        frontal_faces_ratio.append(0.0)

    return frontal_faces_num, frontal_faces_ratio


def shot_change(gray_diff, mag_mu, f_diff, current_shot_duration):
    """
    Decides if a shot change was happened,
    by trying to find big changes between frames.
    """

    gray_diff[gray_diff < 50] = 0
    gray_diff[gray_diff > 50] = 1
    gray_diff_t = gray_diff.sum() \
        / float(gray_diff.shape[0] * gray_diff.shape[1])

    if ((mag_mu > 0.06) and (gray_diff_t > 0.55) and (f_diff[-1] > 0.002) and
            current_shot_duration > 1.1):
        return True
    else:
        return False


def calc_shot_duration(shot_change_times,
                       shot_change_process_indices, shot_durations):
    """Calculates shot duration."""

    shot_avg = 0
    if len(shot_change_times) - 1 > 5:
        for si in range(
                len(shot_change_times) - 1):
            shot_avg += (shot_change_times[si + 1] -
                         shot_change_times[si])
        print('Average shot duration: {}'.format(
                shot_avg / float(len(shot_change_times) - 1)))

    for ccc in range(shot_change_process_indices[-1] -
                     shot_change_process_indices[-2]):
        shot_durations.append(
            shot_change_process_indices[-1] -
            shot_change_process_indices[-2])

    return shot_durations


def windows_display(vis, height, process_mode, v_norm, hist_rgb_ratio,
                    hist_v, hist_s, frontal_faces_num,
                    frontal_faces_ratio, tilt_pan_confidences):
    """
    Displays the processed video and its gray-scaled version.
    Also shows histograms of:
        - Frame saturation
        - Frame value
        - RGB ratio
        - Tilt/Pan confidences
        - Frontal faces number
        - Frontal faces ratio
    """

    plot_width = 150
    plot_width2 = 150
    cv2.imshow('Color', vis)
    cv2.imshow('GrayNorm', v_norm / 256.0)
    cv2.moveWindow('Color', 0, 0)
    cv2.moveWindow('GrayNorm', new_width, 0)

    if process_mode > 0:
        display_histogram(
            np.repeat(hist_rgb_ratio,
                      plot_width2 / hist_rgb_ratio.shape[0]),
            plot_width2,
            height,
            np.max(hist_rgb_ratio),
            'Color Hist')

        display_histogram(
            np.repeat(hist_v,
                      plot_width2 / hist_v.shape[0]),
            plot_width2,
            height,
            np.max(hist_v),
            'Value Hist')

        display_histogram(
            np.repeat(hist_s,
                      plot_width2 / hist_s.shape[0]),
            plot_width2,
            height,
            np.max(hist_s),
            'Sat Hist')

        cv2.moveWindow('Color Hist', 0, height + 70)
        cv2.moveWindow('Value Hist', plot_width2,
                       height + 70)
        cv2.moveWindow('hsv Diff', 2 * plot_width2,
                       height + 70)
        cv2.moveWindow('Sat Hist', 2 * plot_width2,
                       height + 70)

    if process_mode > 1:
        display_histogram(
            np.array(frontal_faces_num),
            plot_width,
            height,
            5,
            'Number of Frontal Faces')

        display_histogram(np.array(frontal_faces_ratio),
                          plot_width,
                          height,
                          1,
                          'Ratio of Frontal Faces')

        display_histogram(
            np.array(tilt_pan_confidences),
            plot_width,
            height,
            50,
            'Tilt Pan Confidences')

        cv2.moveWindow('frontal_faces_num', 0,
                       2 * height + 70)
        cv2.moveWindow('frontal_faces_ratio', plot_width2,
                       2 * height + 70)
        cv2.moveWindow('tilt_pan_confidences',
                       2 * plot_width2,
                       2 * height + 70)

        cv2.waitKey(1)
    return None


def get_features_stats(feature_matrix):
    """
    Calculates statistics on features over time
    and puts them to the feature matrix.
    """

    f_mu = feature_matrix.mean(axis=0)
    f_std = feature_matrix.std(axis=0)
    f_stdmu = feature_matrix.std(axis=0) \
        / (np.median(feature_matrix, axis=0) + 0.0001)
    feature_matrix_sorted_rows = np.sort(feature_matrix, axis=0)
    feature_matrix_sorted_rows_top10 = feature_matrix_sorted_rows[
                                       - int(0.10 *
                                             feature_matrix_sorted_rows.
                                             shape[0])::,
                                       :]
    f_mu10top = feature_matrix_sorted_rows_top10.mean(axis=0)
    features_stats = np.concatenate((f_mu, f_std, f_stdmu, f_mu10top), axis=1)

    return features_stats


def display_time(dur_secs, fps_process, t_process, t_0,
                 duration_secs, duration_str, width, vis):
    """Displays time related strings at the displayed video's window."""

    t_2 = time.time()
    hrs, mins, secs, tenths = seconds_to_time(dur_secs)
    time_str = '{0:02d}:{1:02d}:{2:02d}.{3:02d}'. \
        format(hrs, mins, secs, tenths)
    fps_process = np.append(
        fps_process, plot_step / float(t_2 - t_0))
    t_process = np.append(t_process, 100.0 *
                          float(t_2 - t_0) /
                          (process_step * plot_step))
    if len(fps_process) > 250:
        fps_process_win_avg = np.mean(fps_process[-250:-1])
        t_process_win_avg = np.mean(t_process[-250:-1])
    else:
        fps_process_win_avg = np.mean(fps_process)
        t_process_win_avg = np.mean(t_process)

    hrs, mins, secs, _ = seconds_to_time(
        t_process_win_avg *
        float(duration_secs -
              dur_secs) / 100.0)

    time_remain_str = '{0:02d}:{1:02d}:{2:02d}'. \
        format(hrs, mins, secs)
    string_to_plot = \
        '{0:s}/{1:s} {2:5.1f}fps,{3:2.1f}xR {4:s}'.format(
            time_str, duration_str, fps_process_win_avg,
            100.0 / float(t_process_win_avg), time_remain_str)
    cv2.rectangle(vis, (0, 0), (width, 17),
                  (255, 255, 255), -1)
    cv2.putText(vis, string_to_plot, (20, 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 0))

    return t_2


def process_video(video_path, process_mode, print_flag, save_results):
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
    """

    # ---Initializations-------------------------------------------------------
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
    while True:
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

                print('Shape of feature matrix: {}'.format(
                                                feature_matrix.shape))

                # ---Display features on windows------------------------------
                if (count_process > 2) and (count_process % plot_step == 0) \
                        and print_flag:
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

                    t_0 = t_2
                process_now = False
                img_gray_prev = img_gray

        else:
            break

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
        print('Shape of feature matrix found: {}'.format(
            feature_matrix.shape))
        print('Shape of features\' stats found: {}'.format(
            features_stats.shape))

    if process_mode > 0 and save_results:
        np.savetxt("feature_matrix.csv", feature_matrix, delimiter=",")
        np.savetxt("features_stats.csv", features_stats, delimiter=",")

    return features_stats, feature_matrix


def dir_process_video(dir_name):
    """
    """
    dir_name_no_path = os.path.basename(os.path.normpath(dir_name))

    features_all = np.array([])

    types = ('*.avi', '*.mpeg', '*.mpg', '*.mp4', '*.mkv')
    video_files_list = []
    for files in types:
        video_files_list.extend(glob.glob(os.path.join(dir_name, files)))
    video_files_list = sorted(video_files_list)

    for movieFile in video_files_list:
        print(movieFile)
        [features_stats, feature_matrix] = process_video(movieFile, 2,
                                                         True, False)
        np.save(movieFile + ".npy", feature_matrix)
        if len(features_all) == 0:  # append feature vector
            features_all = features_stats
        else:
            features_all = np.vstack((features_all, features_stats))
    np.save(dir_name_no_path + "_features.npy", features_all)
    np.save(dir_name_no_path + "_video_files_list.npy", video_files_list)
    return features_all, video_files_list


def dirs_process_video(dir_names):
    # feature extraction for each class:
    features = []
    class_names = []
    filenames = []
    for i, d in enumerate(dir_names):
        [f, fn] = dir_process_video(d)
        # if at least one audio file has been found in the provided folder:
        if f.shape[0] > 0:
            features.append(f)
            filenames.append(fn)
            if d[-1] == "/":
                class_names.append(d.split(os.sep)[-2])
            else:
                class_names.append(d.split(os.sep)[-1])
    return features, class_names, filenames


def npy_to_csv(filename_features, filename_names):
    features = np.load(filename_features)
    names = np.load(filename_names)
    fp = open(filename_features.replace(".npy", ".csv"), 'w')
    for i, name in enumerate(names):
        fp.write(os.path.basename(os.path.normpath(name)) + "\t"),
        for f in features[i]:
            fp.write("{0:.6f}\t".format(f))
        fp.write("\n")


def analyze(filename_features, filename_names, first_feature=0,
            last_feature=108, specific_features=[]):
    features = np.load(filename_features)
    names = np.load(filename_names)

    text_file = open("ground_names.txt", "r")
    gt_names = text_file.readlines()
    gt_names = [g.replace("\n", "") for g in gt_names]
    gt_sim = np.load("ground_sim_np")

    # normalize
    mu = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    for i in range(features.shape[0]):
        features[i, :] = (features[i] - mu) / std

    first_pos = []
    second_pos = []
    top10 = []
    top10_second = []

    for i in range(len(names)):  # for each movie
        name_cur = os.path.basename(
            os.path.normpath(names[i])) \
            .replace(".mkv", "") \
            .replace(".mpg", "") \
            .replace(".mp4", "").replace(".avi", "")
        gt_index = gt_names.index(name_cur)
        gt_sim_cur = gt_sim[gt_index, :]
        gt_sim_cur[gt_index] = 0
        gt_sorted = [x for (y, x) in sorted(zip(gt_sim_cur, gt_names),
                                            reverse=True)]

        features_to_use = range(first_feature, last_feature)
        if len(specific_features) > 0:
            features_to_use = specific_features
        else:
            features_to_use = range(first_feature, last_feature)

        # features_to_use = [5, 6, 7, 13, 16, 17, 25, 26, 27, 32, 44, 47]
        d = dist.cdist(
            features[i, features_to_use].reshape(1,
                                                 len(features_to_use)),
            features[:, features_to_use])

        d[0][i] = 100000000
        d = d.flatten()
        # print d.shape, len(N)
        r_sorted = [
            os.path.basename(
                os.path.normpath(x)
            ).replace(".mkv",
                      "").replace(".mpg",
                                  "").replace(".mp4",
                                              "").replace(".avi", "")
            for (y, x) in sorted(zip(d.tolist(), names))]

        first_pos.append(gt_sorted.index(r_sorted[0]) + 1)
        second_pos.append(gt_sorted.index(r_sorted[1]) + 1)
        if r_sorted[0] in gt_sorted[0:10]:
            top10.append(1)
        else:
            top10.append(0)
        if r_sorted[1] in gt_sorted[0:10]:
            top10_second.append(1)
        else:
            top10_second.append(0)

    return np.median(np.array(first_pos)), \
        100 * np.sum(np.array(top10)) / len(top10), \
        np.median(np.array(second_pos)), \
        100 * np.sum(np.array(top10_second)) / len(top10_second)


def analyze_script():
    n_exp = 1000
    all_feature_combinations = []
    med_pos = []
    top10 = []
    med_pos2 = []
    top102 = []

    t_1 = 50
    t_2 = 20
    for n_features in [5, 10, 20, 30, 40, 50, 60, 70]:
        print(n_features)
        for e in range(n_exp):
            features_cur = np.random.permutation(range(108))[0:n_features]
            all_feature_combinations.append(features_cur)
            a1, a2, a3, a4 = analyze(
                "featuresAll.npy",
                "namesAll.npy", 0, 0, features_cur)
            med_pos.append(a1)
            top10.append(a2)
            med_pos2.append(a3)
            top102.append(a4)

    med_pos = np.array(med_pos)
    top10 = np.array(top10)
    med_pos2 = np.array(med_pos2)
    top102 = np.array(top102)

    i_min_pos = np.argmin(med_pos)
    i_max_pos = np.argmax(top10)
    i_min_pos2 = np.argmin(med_pos2)
    i_max_pos2 = np.argmax(top102)

    for i in range(len(top10)):
        if (med_pos[i] < t_1) and (top10[i] > t_2) \
                and (med_pos2[i] < t_1) and (top102[i] > t_2):
            print("{0:.1f}\t{1:.1f}\t{2:.1f}\t{3:.1f}".format(
                med_pos[i], top10[i], med_pos2[i], top102[i]))
            if i == i_min_pos:
                print("BEST med_pos\t")
            else:
                print("-----------\t")
            if i == i_max_pos:
                print("BEST top10\t")
            else:
                print("----------\t")
            if i == i_min_pos2:
                print("BEST med_pos2\t")
            else:
                print("------------\t")
            if i == i_max_pos2:
                print("BEST top102\t")
            else:
                print("-----------\t")

            for f in all_feature_combinations[i]:
                print("{0:d},".format(f))


def main(argv):
    if len(argv) == 3 and argv[1] == "-f":
        process_video(argv[2], 2, True, True)
    if argv[1] == "-d":  # directory
        dir_name = argv[2]
        features_all, video_files_list = dir_process_video(dir_name)
        print(features_all.shape, video_files_list)
    if argv[1] == "evaluate":
        [a, b, a2, b2] = analyze("featuresAll.npy", "namesAll.npy")
        print("First returned result median position {0:.1f}".format(a))
        print("First returned result in top10 {0:.1f} %".format(b))
        print("Second returned result median position {0:.1f}".format(a2))
        print("Second returned result in top10 {0:.1f} %".format(b2))

    if argv[1] == "scriptDebug":
        analyze_script()


if __name__ == '__main__':
    main(sys.argv)
