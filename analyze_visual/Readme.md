# How to run Multi-class 
## Main script: LSTM_pytorch.py

* 3 class
```
python3 LSTM_pytorch.py -v /hand_crafted_features/3_class/Static /hand_crafted_features/3_class/Zoom /hand_crafted_features/3_class/Vertical_and_horizontal_movements 
```
* 4 class
```
python3 LSTM_pytorch.py -v /hand_crafted_features/3_class/Static /hand_crafted_features/3_class/Zoom /hand_crafted_features/3_class/Panoramic /hand_crafted_features/3_class/Tilt
```
* 9 class
```
python3 LSTM_pytorch.py -v /hand_crafted_features/3_class/Static /hand_crafted_features/3_class/Panoramic_lateral /hand_crafted_features/3_class/Vertical_movements /hand_crafted_features/3_class/Handled /hand_crafted_features/3_class/Zoom in /hand_crafted_features/3_class/Travelling_in /hand_crafted_features/3_class/Travelling_out /hand_crafted_features/3_class/Panoramic /hand_crafted_features/3_class/Aerial (51 shots)
```

Generally

* ```LSTM_model``` class includes Sequential with Linear Layers, Dropout etc. 
* ```main``` includes the parameters of the model

In ```main```, there is a dropout parameter for the LSTM. It can be used by removing the [line 392](https://github.com/tyiannak/multimodal_movie_analysis/blob/0f513f8427f67dd9bdb8797b368d78384db7e5f4/analyze_visual/LSTM_pytorch.py#L392/) from comments.

There is a random_state parameter for both "train_test_split" processes in ```data_preparation()``` function.

For each epoch, training and validation losses are calculated, along with a confusion matrix, the accuracy and the f1 (macro averaged). At the end there is a classification report for the test set.

The data of each class can be found at the [hand_crafted_features](https://uthnoc-my.sharepoint.com/personal/apetrogianni_o365_uth_gr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fapetrogianni%5Fo365%5Futh%5Fgr%2FDocuments%2FFEATURES%2Fhand%5Fcrafted%5Ffeatures%2Ezip&parent=%2Fpersonal%2Fapetrogianni%5Fo365%5Futh%5Fgr%2FDocuments%2FFEATURES) folder.



