"""CODE UNDER CONSTRUCTION"""

import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches


class SsdNvidia:

    def __init__(self):

        self.precision = 'fp32'
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                                    'nvidia_ssd', model_math=self.precision)
        self.utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                                    'nvidia_ssd_processing_utils')
        self.classes_to_labels = self.utils.get_coco_object_dictionary()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        print("Using:", self.device)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image, confidence_threshold):
        """image = filename"""

        input_img = self.utils.prepare_input(image)
        input_img = np.expand_dims(input_img, axis=0)

        tensor = self.utils.prepare_tensor(input_img,
                                           self.precision == 'fp16')

        with torch.no_grad():
            detections_batch = self.model(tensor)

        results_per_input = self.utils.decode_results(detections_batch)
        best_results_per_input = [self.utils.pick_best(results,
                                                       confidence_threshold)
                                  for results in results_per_input]

        return best_results_per_input

    def plot_image(self, image, results):

        for image_idx, res in enumerate(results):
            fig, ax = plt.subplots(1)
            # Show original, denormalized image...
            image = image / 2 + 0.5
            ax.imshow(image)
            # ...with detections
            bboxes, classes, confidences = results[image_idx]
            for idx in range(len(bboxes)):
                left, bot, right, top = bboxes[idx]
                x, y, w, h = [val * 300 for val in
                              [left, bot, right - left, top - bot]]
                rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                         edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x, y, "{} {:.0f}%".format(self.classes_to_labels[classes[idx] - 1],
                                                  confidences[idx]*100),
                        bbox=dict(facecolor='white', alpha=0.5))
        plt.show()
        return None

    def display_cv2(self, image, results, window):
        """input = """

        img_cv2 = cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2BGR)

        bboxes, classes, confidences = results[0]
        for idx, box in enumerate(bboxes):
            left, bot, right, top = box
            x, y, w, h = [int(val * 300)
                          for val in [left, bot, right - left, top - bot]]
            img_cv2 = cv2.rectangle(img_cv2, (x, y),
                                    (x + w, y + h), (0,0,255) , 1)
            label = "{} {:.0f}%".format(
                self.classes_to_labels[classes[idx] - 1],
                confidences[idx]*100)
            label_size = cv2.getTextSize(label,
                                         cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)
            x_label = x + label_size[0][0]
            y_label = y - int(label_size[0][1])
            img_cv2 = cv2.rectangle(img_cv2, (x+3, y+3),
                                    (x_label, y_label),
                                    (255, 255, 255), cv2.FILLED)
            img_cv2 = cv2.putText(img_cv2, label, (x, y),
                                  cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow(window, img_cv2)
        cv2.waitKey(0)
        return None

