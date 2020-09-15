"""CODE UNDER CONSTRUCTION"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from os.path import expanduser


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


class SsdNvidia:

    def __init__(self):

        self.tfms = transforms.Compose([
            transforms.Resize(300),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.precision = 'fp32'

        # fix torch hub's code problem with empty frames
        # if the code gets updated, remove these lines
        home_dir = expanduser("~")
        nvidia_dir = home_dir \
            + '/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub'
        a = os.path.exists(nvidia_dir)

        if not a:
            self.model = torch.hub.load(
                'NVIDIA/DeepLearningExamples:torchhub',
                'nvidia_ssd', model_math=self.precision)

            utils_file = nvidia_dir + '/PyTorch/Detection/SSD/src/utils.py'

            with open(utils_file, 'r') as f:
                get_all = f.readlines()

            spaces_str = '    '
            with open(utils_file, 'w') as f:
                for i, line in enumerate(get_all, 1):
                    if i == 196:
                        f.writelines("\n")
                        f.writelines(spaces_str + spaces_str
                                     + "if not bboxes_out:\n")
                        f.writelines(spaces_str + spaces_str + spaces_str
                                     + "return [torch.tensor([]) for _ in range(3)]\n")
                        f.writelines("\n")
                    else:
                        f.writelines(line)
        else:
            self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                                        'nvidia_ssd',
                                        model_math=self.precision)
        # ---end of torch hub's code fixing-----------------------------------

        self.utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                                    'nvidia_ssd_processing_utils')

        self.classes_to_labels = self.utils.get_coco_object_dictionary()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        print("Using:", self.device)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image, confidence_threshold):
        """
        Detects 80 possible objects in an image.
        Args:
            image : an image in the BGR format of OpenCV.
                    No specific dimensions needed.
            confidence_threshold : the threshold that a labels' confidence
            must exceed in order to be counted
        """
        tensor = self.tfms(Image.fromarray(image[:, :, ::-1])).unsqueeze(0).to(self.device)

        with torch.no_grad():
            detections_batch = self.model(tensor)

        bboxes, labels = detections_batch
        if bboxes.nelement() == 0 or labels.nelement() == 0:
            return [torch.tensor([]) for _ in range(3)]

        results = self.utils.decode_results(detections_batch)
        best_results = self.utils.pick_best(results[0], confidence_threshold)
        return best_results

    def display_cv2(self, image, results, window):
        """
        Displays the objects found on a processed image.

        Args:
             image : an image in the BGR format of OpenCV.
                    No specific dimensions needed.
            results : a list of arrays containing the results of
                    the object detection, in the format
                    (bboxes, classes, confidences)
             window : name of the window to display the image
        """
        width, height = image.shape[1], image.shape[0]
        img = cv2.resize(image, (300, 300))
        bboxes, classes, confidences = results

        for idx, box in enumerate(bboxes):
            left, bot, right, top = box
            x, y, w, h = [int(val * 300)
                          for val in [left, bot, right - left, top - bot]]
            img = cv2.rectangle(img, (x, y),
                                    (x + w, y + h), (0,0,255) , 1)
            label = "{} {:.0f}%".format(
                self.classes_to_labels[classes[idx] - 1],
                confidences[idx]*100)
            label_size = cv2.getTextSize(label,
                                         cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)
            x_label = x + label_size[0][0]
            y_label = y - int(label_size[0][1])
            img = cv2.rectangle(img, (x+3, y+3),
                                    (x_label, y_label),
                                    (255, 255, 255), cv2.FILLED)
            img = cv2.putText(img, label, (x, y),
                              cv2.FONT_HERSHEY_COMPLEX,
                              0.5, (0, 0, 0), 1)

        img = cv2.resize(img, (width, height))

        cv2.imshow(window, img)
        return None

    def camera_demo(self):
        """
        Test the ssd model on your camera.
        Just initialize an SsdNvidia object and call camera_demo function.
        """

        capture = cv2.VideoCapture(0)
        fps = capture.get(cv2.CAP_PROP_FPS)
        window_name = 'Object Detection'
        cnt = 0
        while capture.isOpened():
            ret, frame = capture.read()
            if ret:
                if cnt == 0:
                    width, height = frame.shape[1], frame.shape[0]
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, (width, height))

                if cnt % int(fps/10) == 0:
                    detected = self.detect(frame, 0.4)
                    self.display_cv2(frame, detected, window_name)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                cnt += 1
            else:
                capture.release()
                cv2.destroyAllWindows()

        return None


def create_label_map():
    cnt = 1
    tmp_array = np.array([10, 15, 25, 30, 40, 47, 57, 63, 69, 74, 81])
    dictionary = dict()
    dictionary[1] = 1
    for idx, val in enumerate(tmp_array):
        for j in range(cnt + 1, val):
            dictionary[j] = int(idx + 2)
        cnt = j
    return dictionary


def get_object_features(boxes_all, labels_all, confidences_all, which_categories):
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
            sup_label = np.zeros(labels.shape[0])
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
                super_tmp[super_label_map[int(label)] - 1] += 1
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
