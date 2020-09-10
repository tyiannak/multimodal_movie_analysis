"""CODE UNDER CONSTRUCTION"""

import os
import cv2
import torch
from PIL import Image
from torchvision import transforms
from os.path import expanduser


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

        self.utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                                    'nvidia_ssd_processing_utils')

        self.classes_to_labels = self.utils.get_coco_object_dictionary()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        print("Using:", self.device)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image, confidence_threshold):

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
        """input = """
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
