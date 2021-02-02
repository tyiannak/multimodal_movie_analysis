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