# Visual Analysis

Extraction of visual-based features from videos.

## Usage

### Main script: analyze_visual.py

Flags:
* f: extract features from specific file<br/>
    E.g. python3 analyze_visual.py -f \<filename\>
* d: extract features from all files of a directory<br/>
     E.g. python3 analyze_visual.py -d \<directory_name\>   

Basic functions:
* process_video :  extracts features from specific file
* dir_process_video : extracts features from all files of a directory
* dirs_process_video : extracts features from all files of different directories

Please read the docstrings for further information.

#### More details on process_video()
`process_video()` is capable of extracting visual features per frame and for the whole video.
In particular, every 0.2 seconds (i.e. the `process_step` variable defined in `utils.py`), 
the following 88 visual features are extracted:
 * Color - related features (45 features):
    * 8-bin histogram of the red values
    * 8-bin histogram of the green values
    * 8-bin histogram of the blue values
    * 8-bin histogram of the grayscale values
    * 5-bin histogram of the max-by-mean-ratio for each RGB triplet
    * 8-bin histogram of the saturation values
 * Average absolute diff between two successive frames (gray scale), 1 feature
 * Facial features (2 features): the Viola-Jones opencv implementation is used to detect frontal faces and the following features are extracted per frame:
   * number of faces detected
   * average ratio of the faces' bounding boxes areas divided by the total area of the frame
 * Optical-flow related features (3 features): The optical flow is estimated 
  using the Lucas-Kanade method and the following 3 features are extracted:
    * average magnitude of the flow vectors
    * standard deviation of the angles of the flow vectors
    * a hand-crafted feature that measures the possibility that there is a camera tilt movement. 
    This is achieved by measuring a ratio of the magnitude of the flow vectors by the deviation of the angles of the flow vectors. 
 * Current shot duration (1 feature): a basic shot detection method is implemented in this library 
 (see function utils.shot_change()). The length of the shot (in seconds)
 in which each frame belongs to, is used as a feature.
 * Object-related features (36): We use the TODO PANAGIOTIS method for detecting 
 12 categories of objects. For each frame, as soon as the object(s) of each 
 category are detected, three statistics are extracted: number of objects detected,
 average detection confidence and average ratio of the objects' area to the area 
 of the frame. So in total, 3x12=36 object-related features are extracted. 
 The 12 object categories we detect are the following:
    * person
    * vehicle
    * outdoor
    * animal
    * accessory
    * sports
    * kitchen
    * food
    * furniture
    * electronic
    * appliance
    * indoor

In addition, `process_audio()` computes and returns video-level statistics of the 
above non-object features. In particular, mean, std, median by std ratio and top-10 average 
are computed for each of the 52 non-object features for the whole video. 
This results to 52x4 + 36 = 244 feature statistics that describe the whole video.
TODO PANAGIOTIS STH ON THE OBJECT POST PROCESSING HERE

To sum up `process_audio()` returns five arguments: 
 * the 244 feature vector of the whole video (as a numpy array)
 * the 244 corresponding feature statistic names (as a list)
 * the feature matrix of the 88-D feature vectors (one for each frame) as a 
2D numpy matrix, 
 * the list of 88 feature names and 
 * the list of shot changes (in seconds)
 
### Train Shot models
```
python3 train.py -v /media/tyiannak/DATA/Movies/dataset_annotated_2/Panoramic /media/tyiannak/DATA/Movies/dataset_annotated_2/Zoom_in /media/tyiannak/DATA/Movies/dataset_annotated_2/Travelling_in /media/tyiannak/DATA/Movies/dataset_annotated_2/Aerial /media/tyiannak/DATA/Movies/dataset_annotated_2/Tilt /media/tyiannak/DATA/Movies/dataset_annotated_2/Static /media/tyiannak/DATA/Movies/dataset_annotated_2/Vertical_movement /media/tyiannak/DATA/Movies/dataset_annotated_2/Panoramic_lateral /media/tyiannak/DATA/Movies/dataset_annotated_2/Travelling_out /media/tyiannak/DATA/Movies/dataset_annotated_2/Handled -a SVM
```

The following files will be saved to disk:
 * `shot_classifier_SVM.pkl` the classifier
 * `shot_classifier_SVM_scaler.pkl` the scaler
 * `shot_classifier_SVM_results.json` the cross validation results
 * `shot_classifier_conf_mat_SVC().jpg` the confusion matrix of the cross validation procedure

### Run the shots classifier wrapper:
(need to train the model as seen above)

```
python3 wrapper.py -m SVM -i "test/I_m Not There (2007).avi_shot_6768_6774..avi.mp4"
```

(can also give a folder that contains videos in -i)