import warnings
import argparse
import cv2
import glob
import os
import fnmatch
import numpy as np
import torch
import math
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional, cast
from torch import Tensor
from torchvision import transforms, models
from collections import OrderedDict

"""
non_static: 583 shots VS Static: 941 shots

RUN: 
python3 VGG16_features.py -v /media/ubuntu/Seagate/ChromeDownloads/dataset_annotated_4/non_static /media/ubuntu/Seagate/ChromeDownloads/dataset_annotated_4/Static
"""


def parse_arguments():
    """
    Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Create Shot "
                                                 "Classification Dataset")
    parser.add_argument("-v", "--videos_path", required=True, action='append',
                        nargs='+', help="Videos folder path")

    return parser.parse_args()


def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    args = parse_arguments()
    videos_path = args.videos_path

    videos_path = [item for sublist in videos_path for item in sublist]

    model = models.vgg16(pretrained=True)
    # model = models.vgg19(pretrained=True)

    for dir_name in videos_path:
        VGG_extraction_done = fnmatch.filter(os.listdir(dir_name), 'VGG_done.npy')
        if len(VGG_extraction_done) > 0:
            # check if VGG feature extraction is done for this class-folder
            continue

        print("\nVGG feature extraction process for class", os.path.basename(dir_name), "has started. Please wait...")
        types = ('*.avi', '*.mpeg', '*.mpg', '*.mp4', '*.mkv', '*.webm')
        video_files_list = []
        for files in types:
            video_files_list.extend(glob.glob(os.path.join(dir_name, files)))
        video_files_list = sorted(video_files_list)

        for movieFile in video_files_list:
            h = model.classifier[5].register_forward_hook(get_features('feats'))

            # scaling the frame's RGB pixel intensities by subtracting the mean
            # and then dividing by the standard deviation
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            features = {}
            count = 0
            next_process_stamp = 0.0
            PREDS = []
            FEATS = []
            capture = cv2.VideoCapture(movieFile)
            # frameRate = capture.get(5)

            while capture.isOpened():
                # get frame
                frameID = capture.get(1)
                ret, frame = capture.read()

                # ---Begin processing-------------------------------------------------
                if ret:
                    count += 1
                    if (count % 5) == 1:
                        frame = cv2.resize(frame, (224, 224))
                        img = preprocess(frame)
                        img = img.unsqueeze(0)
                        preds = model(img)
                        PREDS.append(preds.detach().cpu().numpy())
                        FEATS.append(features['feats'].cpu().numpy())
                else:
                    capture.release()
                    cv2.destroyAllWindows()

            PREDS = np.concatenate(PREDS)
            FEATS = np.concatenate(FEATS)
            h.remove()
            #print('- preds shape:', PREDS.shape)
            #print('- feats shape:', FEATS.shape)
            #print(PREDS, "\n")
            #print(FEATS)
            np.save(movieFile + "_LAST_VGG.npy", PREDS)
            np.save(movieFile + "_FEATS_VGG.npy", FEATS)

        np.save(os.path.join(dir_name + "/" + "VGG_done.npy"), "feature extraction done")
