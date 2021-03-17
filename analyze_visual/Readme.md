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
 * Object-related features (36): We use the Single Shot Multibox Detector method for detecting 
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

In addition, `process_video()` computes and returns six (6) video-level statistics of the 
above non-object features. In particular, mean, std, median by std ratio, top-10 percentile, 
mean of the delta features and std of the delta features are computed for each of the 52 non-object features for the whole video. 
As for the object detection, the frame-level predictions are post processed under 
local time widnows with two different ways: 
(i) the object frame-level confidences are smoothed across time windows in 
order to increase the accuracy of the predictions and 
(ii)  every object that is not present to at least a minimum number (threshold)
 of subsequent frames, is excluded from the final feature vector. However, 
 this smoothing procedure is the only post-processing performed on the object-related features: 
 no other statistics are extracted for the whole video, other than the object features' simple averages.
This process therefore results to 52x6 + 36 = 348 feature statistics that describe the whole video.

To sum up `process_video()` returns five arguments: 
 * the 244 feature vector of the whole video (as a numpy array)
 * the 244 corresponding feature statistic names (as a list)
 * the feature matrix of the 88-D feature vectors (one for each frame) as a 
2D numpy matrix, 
 * the list of 88 feature names and 
 * the list of shot changes (in seconds)

Currently the 88 feature names per frame are:
```
['hist_r0', 'hist_r1', 'hist_r2', 'hist_r3', 'hist_r4', 'hist_r5', 'hist_r6', 'hist_r7', 'hist_g0', 'hist_g1', 'hist_g2', 'hist_g3', 'hist_g4', 'hist_g5', 'hist_g6', 'hist_g7', 'hist_b0', 'hist_b1', 'hist_b2', 'hist_b3', 'hist_b4', 'hist_b5', 'hist_b6', 'hist_b7', 'hist_v0', 'hist_v1', 'hist_v2', 'hist_v3', 'hist_v4', 'hist_v5', 'hist_v6', 'hist_v7', 'hist_rgb0', 'hist_rgb1', 'hist_rgb2', 'hist_rgb3', 'hist_rgb4', 'hist_s0', 'hist_s1', 'hist_s2', 'hist_s3', 'hist_s4', 'hist_s5', 'hist_s6', 'hist_s7', 'frame_value_diff', 'frontal_faces_num', 'fronatl_faces_ratio', 'tilt_pan_confidences', 'mag_mean', 'mag_std', 'shot_durations', 'person_num', 'vehicle_num', 'outdoor_num', 'animal_num', 'accessory_num', 'sports_num', 'kitchen_num', 'food_num', 'furniture_num', 'electronic_num', 'appliance_num', 'indoor_num', 'person_confidence', 'vehicle_confidence', 'outdoor_confidence', 'animal_confidence', 'accessory_confidence', 'sports_confidence', 'kitchen_confidence', 'food_confidence', 'furniture_confidence', 'electronic_confidence', 'appliance_confidence', 'indoor_confidence', 'person_area_ratio', 'vehicle_area_ratio', 'outdoor_area_ratio', 'animal_area_ratio', 'accessory_area_ratio', 'sports_area_ratio', 'kitchen_area_ratio', 'food_area_ratio', 'furniture_area_ratio', 'electronic_area_ratio', 'appliance_area_ratio', 'indoor_area_ratio']
```
while the 244 video-level feature names are :
```
['mean_hist_r0', 'mean_hist_r1', 'mean_hist_r2', 'mean_hist_r3', 'mean_hist_r4', 'mean_hist_r5', 'mean_hist_r6', 'mean_hist_r7', 'mean_hist_g0', 'mean_hist_g1', 'mean_hist_g2', 'mean_hist_g3', 'mean_hist_g4', 'mean_hist_g5', 'mean_hist_g6', 'mean_hist_g7', 'mean_hist_b0', 'mean_hist_b1', 'mean_hist_b2', 'mean_hist_b3', 'mean_hist_b4', 'mean_hist_b5', 'mean_hist_b6', 'mean_hist_b7', 'mean_hist_v0', 'mean_hist_v1', 'mean_hist_v2', 'mean_hist_v3', 'mean_hist_v4', 'mean_hist_v5', 'mean_hist_v6', 'mean_hist_v7', 'mean_hist_rgb0', 'mean_hist_rgb1', 'mean_hist_rgb2', 'mean_hist_rgb3', 'mean_hist_rgb4', 'mean_hist_s0', 'mean_hist_s1', 'mean_hist_s2', 'mean_hist_s3', 'mean_hist_s4', 'mean_hist_s5', 'mean_hist_s6', 'mean_hist_s7', 'mean_frame_value_diff', 'mean_frontal_faces_num', 'mean_fronatl_faces_ratio', 'mean_tilt_pan_confidences', 'mean_mag_mean', 'mean_mag_std', 'mean_shot_durations', 'std_hist_r0', 'std_hist_r1', 'std_hist_r2', 'std_hist_r3', 'std_hist_r4', 'std_hist_r5', 'std_hist_r6', 'std_hist_r7', 'std_hist_g0', 'std_hist_g1', 'std_hist_g2', 'std_hist_g3', 'std_hist_g4', 'std_hist_g5', 'std_hist_g6', 'std_hist_g7', 'std_hist_b0', 'std_hist_b1', 'std_hist_b2', 'std_hist_b3', 'std_hist_b4', 'std_hist_b5', 'std_hist_b6', 'std_hist_b7', 'std_hist_v0', 'std_hist_v1', 'std_hist_v2', 'std_hist_v3', 'std_hist_v4', 'std_hist_v5', 'std_hist_v6', 'std_hist_v7', 'std_hist_rgb0', 'std_hist_rgb1', 'std_hist_rgb2', 'std_hist_rgb3', 'std_hist_rgb4', 'std_hist_s0', 'std_hist_s1', 'std_hist_s2', 'std_hist_s3', 'std_hist_s4', 'std_hist_s5', 'std_hist_s6', 'std_hist_s7', 'std_frame_value_diff', 'std_frontal_faces_num', 'std_fronatl_faces_ratio', 'std_tilt_pan_confidences', 'std_mag_mean', 'std_mag_std', 'std_shot_durations', 'stdmean_hist_r0', 'stdmean_hist_r1', 'stdmean_hist_r2', 'stdmean_hist_r3', 'stdmean_hist_r4', 'stdmean_hist_r5', 'stdmean_hist_r6', 'stdmean_hist_r7', 'stdmean_hist_g0', 'stdmean_hist_g1', 'stdmean_hist_g2', 'stdmean_hist_g3', 'stdmean_hist_g4', 'stdmean_hist_g5', 'stdmean_hist_g6', 'stdmean_hist_g7', 'stdmean_hist_b0', 'stdmean_hist_b1', 'stdmean_hist_b2', 'stdmean_hist_b3', 'stdmean_hist_b4', 'stdmean_hist_b5', 'stdmean_hist_b6', 'stdmean_hist_b7', 'stdmean_hist_v0', 'stdmean_hist_v1', 'stdmean_hist_v2', 'stdmean_hist_v3', 'stdmean_hist_v4', 'stdmean_hist_v5', 'stdmean_hist_v6', 'stdmean_hist_v7', 'stdmean_hist_rgb0', 'stdmean_hist_rgb1', 'stdmean_hist_rgb2', 'stdmean_hist_rgb3', 'stdmean_hist_rgb4', 'stdmean_hist_s0', 'stdmean_hist_s1', 'stdmean_hist_s2', 'stdmean_hist_s3', 'stdmean_hist_s4', 'stdmean_hist_s5', 'stdmean_hist_s6', 'stdmean_hist_s7', 'stdmean_frame_value_diff', 'stdmean_frontal_faces_num', 'stdmean_fronatl_faces_ratio', 'stdmean_tilt_pan_confidences', 'stdmean_mag_mean', 'stdmean_mag_std', 'stdmean_shot_durations', 'mean10top_hist_r0', 'mean10top_hist_r1', 'mean10top_hist_r2', 'mean10top_hist_r3', 'mean10top_hist_r4', 'mean10top_hist_r5', 'mean10top_hist_r6', 'mean10top_hist_r7', 'mean10top_hist_g0', 'mean10top_hist_g1', 'mean10top_hist_g2', 'mean10top_hist_g3', 'mean10top_hist_g4', 'mean10top_hist_g5', 'mean10top_hist_g6', 'mean10top_hist_g7', 'mean10top_hist_b0', 'mean10top_hist_b1', 'mean10top_hist_b2', 'mean10top_hist_b3', 'mean10top_hist_b4', 'mean10top_hist_b5', 'mean10top_hist_b6', 'mean10top_hist_b7', 'mean10top_hist_v0', 'mean10top_hist_v1', 'mean10top_hist_v2', 'mean10top_hist_v3', 'mean10top_hist_v4', 'mean10top_hist_v5', 'mean10top_hist_v6', 'mean10top_hist_v7', 'mean10top_hist_rgb0', 'mean10top_hist_rgb1', 'mean10top_hist_rgb2', 'mean10top_hist_rgb3', 'mean10top_hist_rgb4', 'mean10top_hist_s0', 'mean10top_hist_s1', 'mean10top_hist_s2', 'mean10top_hist_s3', 'mean10top_hist_s4', 'mean10top_hist_s5', 'mean10top_hist_s6', 'mean10top_hist_s7', 'mean10top_frame_value_diff', 'mean10top_frontal_faces_num', 'mean10top_fronatl_faces_ratio', 'mean10top_tilt_pan_confidences', 'mean10top_mag_mean', 'mean10top_mag_std', 'mean10top_shot_durations', 'person_freq', 'vehicle_freq', 'outdoor_freq', 'animal_freq', 'accessory_freq', 'sports_freq', 'kitchen_freq', 'food_freq', 'furniture_freq', 'electronic_freq', 'appliance_freq', 'indoor_freq', 'person_mean_confidence', 'vehicle_mean_confidence', 'outdoor_mean_confidence', 'animal_mean_confidence', 'accessory_mean_confidence', 'sports_mean_confidence', 'kitchen_mean_confidence', 'food_mean_confidence', 'furniture_mean_confidence', 'electronic_mean_confidence', 'appliance_mean_confidence', 'indoor_mean_confidence', 'person_mean_area_ratio', 'vehicle_mean_area_ratio', 'outdoor_mean_area_ratio', 'animal_mean_area_ratio', 'accessory_mean_area_ratio', 'sports_mean_area_ratio', 'kitchen_mean_area_ratio', 'food_mean_area_ratio', 'furniture_mean_area_ratio', 'electronic_mean_area_ratio', 'appliance_mean_area_ratio', 'indoor_mean_area_ratio']
```
 
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
