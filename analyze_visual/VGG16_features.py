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

python3 VGG16_features.py -v /media/ubuntu/Seagate/ChromeDownloads/VGG_AVG_POOL/Non_Static /media/ubuntu/Seagate/ChromeDownloads/VGG_AVG_POOL/Static
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


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
        # Convert the image into one-dimensional vector
        #self.flatten = nn.Flatten()


    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        out = self.pooling(out)
        #out = self.flatten(out)

        return out


# HELPER FUNCTION FOR FEATURE EXTRACTION
def get_features(name):
    #  The 'name' argument in get_features() specifies the dictionary key
    #  under which we will store our intermediate activations.
    def hook(model, input, output):
        # copies the layer outputs, sends them to CPU
        # and saves them to a dictionary object called 'features'
        features[name] = output.detach()
    return hook

# FEATURE EXTRACTION FROM MAX AVG POOL LAYER after convolutional layer
if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    args = parse_arguments()
    videos_path = args.videos_path

    videos_path = [item for sublist in videos_path for item in sublist]

    #model = models.vgg16(pretrained=True)
    #print("features: ", model)
    #modified_pretrained = nn.Sequential(*list(model.features.children())[:32])
    #modules = list(model.children())[:-1]
    #print(model.features[30])
    #vgg16_model = nn.Sequential(*modules)
    #print(vgg16_model)

    model = models.vgg16(pretrained=True)
    new_model = FeatureExtractor(model)


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
            # we can access the VGG-16 classifier with model.classifier, which is an 6-layer array
            # the hook can be applied to any layer of the neural network

            #h = vgg16_model.register_forward_hook(get_features('feats'))

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

            f = []

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

                        feature_extr = new_model(img)
                        feature_extr = feature_extr.reshape(-1,7,512)
                        feature_extr = feature_extr.mean(axis=0).mean(axis=0)
                        feature_extr =feature_extr.cpu().detach().numpy()
                        f.append(feature_extr)

                else:
                    capture.release()
                    cv2.destroyAllWindows()

            features_avg = np.asarray(f)

            np.save(movieFile + "_AVG_POOLING_VGG.npy", features_avg)

        np.save(os.path.join(dir_name + "/" + "VGG_done.npy"), "feature extraction done")
