
import cv2
import time
import numpy as np

# process and plot related parameters:
new_width = 500
process_step = 0.2
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
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, (width, height))

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


def rect_area(rect):
    return rect[0] * rect[1] * rect[2] * rect[3]


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

        tilt_pan_confidence = np.mean(mags) / np.sqrt(std + 0.00000001)
        tilt_pan_confidence = tilt_pan_confidence[0]
        # TODO:
        # CHECK PANCONFIDENCE
        # SAME FOR ZOOM AND OTHER CAMERA EFFECTS

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

    count = 0
    if ((mag_mu > 0.08)):
        count += 1
    if (gray_diff_t > 0.65):
        count += 1
    if (f_diff[-1] > 0.02):
        count += 1

    if (count>=2) and (current_shot_duration > 1.1):
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

    cv2.imshow('Frame Flow', vis)
    cv2.moveWindow('Frame Flow', 0, 0)

    if process_mode > 0:
        display_histogram(
            np.repeat(hist_rgb_ratio,
                      plot_width / hist_rgb_ratio.shape[0]),
            plot_width,
            height,
            np.max(hist_rgb_ratio),
            'Color Hist')

        display_histogram(
            np.repeat(hist_v,
                      plot_width / hist_v.shape[0]),
            plot_width,
            height,
            np.max(hist_v),
            'Value Hist')

        display_histogram(
            np.repeat(hist_s,
                      plot_width / hist_s.shape[0]),
            plot_width,
            height,
            np.max(hist_s),
            'Sat Hist')

        cv2.moveWindow('Color Hist', 0, height + 90)
        cv2.moveWindow('Value Hist', plot_width,
                       height + 90)
        cv2.moveWindow('hsv Diff', 2 * plot_width,
                       height + 90)
        cv2.moveWindow('Sat Hist',  3 * plot_width,
                       height + 90)

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
            1,
            'Tilt Pan Confidences')

        cv2.moveWindow('frontal_faces_num', 4 * plot_width,
                       height + 90)
        cv2.moveWindow('frontal_faces_ratio', 5 * plot_width,
                       height + 90)
        cv2.moveWindow('tilt_pan_confidences',
                       6 * plot_width,
                       height + 90)
    return None


def get_features_stats(feature_matrix):
    """
    Calculates statistics on features over time
    and puts them to the feature stats vector.
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
    features_stats = np.squeeze(np.asarray(features_stats))

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


def get_features_names(process_mode, which_object_categories):
    feature_names = []
    hist_r = 'hist_r'
    hist_g = 'hist_g'
    hist_b = 'hist_b'
    hist_v = 'hist_v'
    hist_rgb_ratio = 'hist_rgb'
    hist_s = 'hist_s'
    for i in range(0, 8):
        feature_names.append(hist_r + '{}'.format(i))
    for i in range(0, 8):
        feature_names.append(hist_g + '{}'.format(i))
    for i in range(0, 8):
        feature_names.append(hist_b + '{}'.format(i))
    for i in range(0, 8):
        feature_names.append(hist_v + '{}'.format(i))
    for i in range(0, 5):
        feature_names.append(hist_rgb_ratio + '{}'.format(i))
    for i in range(0, 8):
        feature_names.append(hist_s + '{}'.format(i))
    frame_value_diff = 'frame_value_diff'
    feature_names.append(frame_value_diff)


    if process_mode > 1:
        frontal_faces_num = 'frontal_faces_num'
        feature_names.append(frontal_faces_num)
        frontal_faces_ratio = 'fronatl_faces_ratio'
        feature_names.append(frontal_faces_ratio)
        tilt_pan_confidences = 'tilt_pan_confidences'
        feature_names.append(tilt_pan_confidences)
        mag_mean = 'mag_mean'
        feature_names.append(mag_mean)
        mag_std = 'mag_std'
        feature_names.append(mag_std)
        shot_durations = 'shot_durations'
        feature_names.append(shot_durations)

        feature_stats_names = ['mean_' + name for name in feature_names]
        feature_stats_names += ['std_' + name for name in feature_names]
        feature_stats_names += ['stdmean_' + name for name in feature_names]
        feature_stats_names += ['mean10top_' + name for name in feature_names]

        if which_object_categories > 0:
            category_names = ['person', 'vehicle', 'outdoor', 'animal',
                                    'accessory', 'sports', 'kitchen', 'food',
                                    'furniture', 'electronic', 'appliance',
                                    'indoor']

        else:
            with open("category_names.txt", encoding="utf-8") as file:
                category_names = [l.rstrip("\n") for l in file]

        for category in category_names:
            feature_names.append(category + '_num')
            feature_stats_names.append(category + '_freq')

        for category in category_names:
            feature_names.append(category + '_confidence')
            feature_stats_names.append(category + '_mean_confidence')

        for category in category_names:
            feature_names.append(category + '_area_ratio')
            feature_stats_names.append(category + '_mean_area_ratio')

    else:
        feature_stats_names = ['mean_' + name for name in feature_names]
        feature_stats_names += ['std_' + name for name in feature_names]
        feature_stats_names += ['stdmean_' + name for name in feature_names]
        feature_stats_names += ['mean10top_' + name for name in feature_names]

    return feature_names, feature_stats_names


def npy_to_csv(filename_features, filename_names):
    features = np.load(filename_features)
    names = np.load(filename_names)
    fp = open(filename_features.replace(".npy", ".csv"), 'w')
    for i, name in enumerate(names):
        fp.write(os.path.basename(os.path.normpath(name)) + "\t"),
        for f in features[i]:
            fp.write("{0:.6f}\t".format(f))
        fp.write("\n")

