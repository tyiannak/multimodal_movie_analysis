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




