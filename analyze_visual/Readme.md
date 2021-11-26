# How to run binary
## Main script: LSTM_pytorch.py

```
python3 LSTM_pytorch.py -v /hand_crafted_features/3_class/Non_Static /hand_crafted_features/3_class/Static
```
Generally

* ```LSTM_model``` class includes Sequential with Linear Layers, Dropout etc. 
* ```main``` includes the parameters of the model

In ```main```, there is a dropout parameter for the LSTM. It can be used by removing the [line 399](https://github.com/tyiannak/multimodal_movie_analysis/blob/41ac63532194d24d0c32251d3a228bdc5381b34b/analyze_visual/LSTM_pytorch.py#L399) from comments.

If you wish to run the experiment for a specific random_state, you must add the random_state parameter at both "train_test_split" processes (in ```data_preparation()``` function).

For each epoch, training and validation losses are calculated, along with a confusion matrix, the accuracy and the f1 (macro averaged). At the end there is a classification report for the test set.

The data of each class can be found at the [hand_crafted_features](https://uthnoc-my.sharepoint.com/personal/apetrogianni_o365_uth_gr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fapetrogianni%5Fo365%5Futh%5Fgr%2FDocuments%2FFEATURES%2Fhand%5Fcrafted%5Ffeatures%2Ezip&parent=%2Fpersonal%2Fapetrogianni%5Fo365%5Futh%5Fgr%2FDocuments%2FFEATURES) folder.
