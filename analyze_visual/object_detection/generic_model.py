import sys
import os
import urllib.request
import cv2
import torch
from PIL import Image
from torchvision import transforms
from os.path import expanduser

def _download_checkpoint(checkpoint, force_reload):
    model_dir = os.path.join(torch.hub._get_torch_home(), 'checkpoints')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    ckpt_file = os.path.join(model_dir, os.path.basename(checkpoint))
    if not os.path.exists(ckpt_file) or force_reload:
        sys.stderr.write('Downloading checkpoint from {}\n'.format(checkpoint))
        urllib.request.urlretrieve(checkpoint, ckpt_file)
    return ckpt_file



class SsdNvidia:
    """
    Class that uses NVIDIA's ssd model
    (https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD)
    ,as it's implemented on NVIDIA's repo torchhub branch,
    to detect 80 object categories.

    NOTE: The torchhub branch on NVIDIA's repo contains a problem
    when empty frames are given as input to the model. The below code fixes
    this issue. If the branch gets updated remove the modification lines.
    """

    def __init__(self):

        self.tfms = transforms.Compose([
            transforms.Resize(300),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.precision = 'fp32'

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        print("Using:", self.device)
        checkpoint_str =\
            'https://api.ngc.nvidia.com/v2/models/nvidia/ssdpyt_fp32/versions/1/files/nvidia_ssdpyt_fp32_20190225.pt'

        # fix torch hub's code problem with empty frames
        # !!!if the code gets updated, remove these lines!!!
        home_dir = expanduser("~")
        nvidia_dir = home_dir \
            + '/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub'
        a = os.path.exists(nvidia_dir)

        if not a:
            self.model = torch.hub.load(
                'NVIDIA/DeepLearningExamples:torchhub',
                'nvidia_ssd', model_math=self.precision,
                pretrained=False)
            ckpt_file = _download_checkpoint(checkpoint_str, force_reload=False)
            if self.use_cuda:
                ckpt = torch.load(ckpt_file)
            else:
                ckpt = torch.load(ckpt_file, map_location=lambda storage, loc: storage)

            ckpt = ckpt['model']
            self.model.load_state_dict(ckpt)

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
            self.model = torch.hub.load(
                'NVIDIA/DeepLearningExamples:torchhub',
                'nvidia_ssd', model_math=self.precision,
                pretrained=False)
            ckpt_file = _download_checkpoint(checkpoint_str, force_reload=False)
            if self.use_cuda:
                ckpt = torch.load(ckpt_file)
            else:
                ckpt = torch.load(ckpt_file, map_location=lambda storage, loc: storage)

            ckpt = ckpt['model']
            self.model.load_state_dict(ckpt)

        # ---end of torch hub's code fixing-----------------------------------

        self.utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                                    'nvidia_ssd_processing_utils')

        self.classes_to_labels = self.utils.get_coco_object_dictionary()
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
