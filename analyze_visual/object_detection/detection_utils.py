import copy
import numpy as np
from utils import rect_area
from utils import intersect_rectangles

super_categories = {1: 'person',
                    2: 'vehicle',
                    3: 'outdoor',
                    4: 'animal',
                    5: 'accessory',
                    6: 'sports',
                    7: 'kitchen',
                    8: 'food',
                    9: 'furniture',
                    10: 'electronic',
                    11: 'appliance',
                    12: 'indoor'
                    }


def create_label_map():
    """
    Creates mapping between labels and super-labels.

    Returns a dictionary of the mapping.
    """

    cnt = 1
    tmp_array = np.array([10, 15, 25, 30, 40, 47, 57, 63, 69, 74, 81])
    dictionary = dict()
    dictionary[1] = 1
    for idx, val in enumerate(tmp_array):
        for j in range(cnt + 1, val):
            dictionary[j] = int(idx + 2)
        cnt = j
    return dictionary


def get_object_features(labels_all, confidences_all,
                        boxes_all, which_categories):

    """
    Extracts video features from objects detected.

    Args:
        - boxes_all (list): list of arrays containing the normalized
                    coordinates of every box for every frame
        - labels_all (list): list of arrays containing the labels of
                    every object detected for every frame
        - confidences_all (list): list of arrays containing the confidences of
                    every object detected for every frame
        - which_categories int: values
                    0: returns features for all 80 categories
                    1: returns features for 12 super categories
                    2: returns features for both 80 and 12 categories

    Returns:
        - labels_freq (array like): frequency of every label per frame
                    E.g.: 4 means that this label averages 4 objects per frame
        - labels_avg_confidence (array like): the average confidence of every
                    object detected
        - labels_area_ratio (array like): the average area occupied by the
                    labels per frame
                    E.g.: 0.4 means that this label occupies 40% of
                    the area per frame
    """

    if not labels_all:
        return None
    frame_area = 300 * 300
    tmp = np.zeros(80)
    labels_freq = np.zeros(80)
    labels_area_ratio = np.zeros(80)
    if which_categories > 0:
        super_label_map = create_label_map()
        super_labels_all = []
        super_tmp = np.zeros(12)
        super_labels_freq = np.zeros(12)
        super_labels_area_ratio = np.zeros(12)

        for i, labels in enumerate(labels_all):
            sup_label = np.zeros(len(labels))
            for j, label in enumerate(labels):
                sup_label[j] = super_label_map[label]
            super_labels_all.append(sup_label)

    for i, labels in enumerate(labels_all):
        for j, label in enumerate(labels):
            left, bot, right, top = boxes_all[i][j]
            x, y, w, h = [int(val * 300)
                          for val in [left, bot, right - left, top - bot]]
            labels_freq[label - 1] += 1
            tmp[label - 1] += confidences_all[i][j]
            labels_area_ratio[label - 1] += (w * h) / float(frame_area)
            if which_categories > 0:
                super_labels_freq[super_label_map[int(label)] - 1] += 1
                super_tmp[super_label_map[int(label)] - 1] += confidences_all[i][j]
                super_labels_area_ratio[super_label_map[int(label)] - 1] +=\
                    (w * h) / float(frame_area)

    labels_avg_confidence = [tmp[idx] / freq if freq > 0 else 0
                             for idx, freq in enumerate(labels_freq)]
    labels_avg_confidence = np.asarray(labels_avg_confidence)
    labels_freq = labels_freq / len(labels_all)
    labels_area_ratio = labels_area_ratio / len(labels_all)

    if which_categories > 0:
        super_labels_avg_confidence = [super_tmp[idx] / freq
                                       if freq > 0 else 0
                                       for idx, freq in enumerate(
                                        super_labels_freq)]
        super_labels_avg_confidence = np.asarray(super_labels_avg_confidence)
        super_labels_freq = super_labels_freq / len(super_labels_all)
        super_labels_area_ratio = super_labels_area_ratio / \
            len(super_labels_all)

        if which_categories > 1:
            return (labels_freq, labels_avg_confidence, labels_area_ratio),\
                   (super_labels_freq, super_labels_avg_confidence,
                    super_labels_area_ratio)
        else:
            return (), (super_labels_freq, super_labels_avg_confidence,
                        super_labels_area_ratio)
    else:
        return (labels_freq, labels_avg_confidence, labels_area_ratio), ()


def detect_movement(labels, boxes, confidences,
                    overlap_threshold, max_frames):
    """
    Verifies the same object across a number of adjacent frames.
    If an object is not in all adjacent frames, it gets excluded.

    Args:
        labels (list):
            A list of all object labels for every adjacent frame.
        boxes (list):
            A list of all object boxes for every adjacent frame.
        confidences (list):
            A list of all object confidences for every adjacent frame.
        overlap_threshold (float):
            The minimum acceptable intersection rate for two rectangles
            of the same object, across different frames.
        max_frames (int):
            The number of adjacent frames to search.


    Returns:
        label_indexes (dict of dicts):
            A dictionary that contains one key for every label.
            The value of this key is another dictionary whose key
            equals the label. Under this label lies a list of lists
            that contains all the confidences, boxes and indexes
            (i.e. indexes of the original labels list)
            of this specific object.
    """

    label_indexes = {}
    for idx, label in enumerate(labels[0]):
        tmp_dict = {label: [[idx], [boxes[0][idx]], [confidences[0][idx]]]}
        label_indexes[idx] = tmp_dict
    for idx, frame in enumerate(labels):
        if idx == 0:
            continue
        for j, label1 in enumerate(frame):
            for k, label2 in enumerate(labels[0]):
                if label1 == label2:
                    inter_ratio = intersect_rectangles(
                                boxes[0][k], boxes[idx][j])
                    if inter_ratio >= overlap_threshold and inter_ratio <= 1:

                        label_indexes[k][label2][0].append(j)
                        label_indexes[k][label2][1].append(boxes[idx][j])
                        label_indexes[k][label2][2].append(confidences[idx][j])

    for label_index, values in list(label_indexes.items()):
        tmp = list(values.values())[0]
        if len(tmp[0]) < max_frames:
            del label_indexes[label_index]

    return label_indexes


def find_smaller_box(boxes):
    """
    Takes a list of coordinates for different rectangles
    and returns the smaller rectangle.
    """

    min_area = rect_area(boxes[0])
    smaller_box = boxes[0]
    for box in boxes[1:]:
        area = rect_area(box)
        if area < min_area:
            min_area = area
            smaller_box = box

    return smaller_box


def smooth_confidences(label_indexes, mean_confidence_threshold):
    """
    Smooths the confidences of specific labels.

    Args:
        label_indexes (dict of dicts):
            Produced by detect_movement function.
            A dictionary that contains one key for every label.
            The value of this key is another dictionary whose key
            equals the label. Under this label lies a list of lists
            that contains all the confidences, boxes and indexes
            (i.e. indexes of the original labels list)
            of this specific object.

        mean_confidence_threshold (float):
            The minimum accepted smoothed confidence.

    Returns:
        out_labels (list): A list of all smoothed object labels.
        out_boxes (list): A list of all smoothed object boxes.
        out_confidences (list): A list of all smoothed object confidences.
    """

    out_labels = []
    out_boxes = []
    out_confidences = []
    for label_index, values in label_indexes.items():
        label = list(values.keys())[0]
        tmp = list(values.values())[0]
        mu = sum(tmp[2]) / len(tmp[2])
        if mu >= mean_confidence_threshold:
            out_labels.append(label)
            out_confidences.append(mu)
            smaller_box = find_smaller_box(tmp[1])
            out_boxes.append(smaller_box)

    return out_labels, out_boxes, out_confidences


def group_frames(labels_all, confidences_all, boxes_all, max_frames):
    """
    Groups a number of adjacent frames, according to max_frames variable.
    """

    grouped_frame_labels = []
    grouped_frame_confidences = []
    grouped_frame_boxes = []
    length = len(labels_all)
    for i, labels in enumerate(labels_all):
        if length - i >= max_frames:
            frames_labels = labels_all[i:i+max_frames]
            frames_confidences = copy.deepcopy(confidences_all[i:i+max_frames])
            frames_boxes = copy.deepcopy(boxes_all[i:i+max_frames])
        elif i + 1 == length:
            break
        else:
            frames_labels = labels_all[i:-1]
            frames_confidences = copy.deepcopy(confidences_all[i:-1])
            frames_boxes = copy.deepcopy(boxes_all[i:-1])

        grouped_frame_labels.append(frames_labels)
        grouped_frame_confidences.append(frames_confidences)
        grouped_frame_boxes.append(frames_boxes)

    return grouped_frame_labels, grouped_frame_confidences, grouped_frame_boxes


def smooth_object_confidence(labels_all, confidences_all,
                             boxes_all, overlap_threshold,
                             mean_confidence_threshold, max_frames):
    """
    Smooths the objects' confidences across a given number of frames
    and returns the oblects for which the smoothed confidences exceed
    the confidence threshold.

    Args:
        labels_all (list):
            A list of all object labels for every frame.
        confidences_all (list):
            A list of all object confidences for every frame.
        boxes_all (list):
            A list of all object boxes for every frame.
        overlap_threshold (float):
            The minimum acceptable intersection rate for two rectangles
            of the same object, across different frames.
        mean_confidence_threshold (float):
            The minimum accepted smoothed confidence
        max_frames (int):
            The number of adjacent frames to apply the smoothing.

    Returns:

        out_labels (list): A list of all smoothed object labels for every frame.
        out_boxes (list): A list of all smoothed object boxes for every frame.
        out_confidences (list): A list of all smoothed object confidences for every frame.

    NOTE:
        For input labels across n frames, the output smoothed labels
        will extend across n - max_frame + 1 frames.
    """

    out_labels = []
    out_boxes = []
    out_confidences = []
    grouped_frame_labels, grouped_frame_confidences, grouped_frame_boxes =\
        group_frames(labels_all, confidences_all, boxes_all, max_frames)

    for idx, frames_labels in enumerate(grouped_frame_labels):
        boxes = grouped_frame_boxes[idx]
        confidences = grouped_frame_confidences[idx]
        label_indexes = detect_movement(
            frames_labels, boxes, confidences, overlap_threshold, max_frames)
        frame_labels, frame_boxes, frame_confidences =\
            smooth_confidences(label_indexes,  mean_confidence_threshold)
        out_labels.append(frame_labels)
        out_boxes.append(frame_boxes)
        out_confidences.append(frame_confidences)

    return out_labels, out_boxes, out_confidences
