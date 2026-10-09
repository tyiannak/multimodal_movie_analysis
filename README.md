# multimodal_movie_analysis

> Developed at [**MagCIL**](https://magcil.github.io), the Multimodal Analysis Group of the Institute of Informatics and Telecommunications, [NCSR "Demokritos"](https://www.demokritos.gr/), Athens, Greece.

## Audio
To analyze a movie in terms of its auditory content, do the following:
```
cd analyze_audio
python3 analyze_audio.py -f movie.wav
```

Note: You will need to create a folder in `analyze_audio/segment_models` where 
you will store your audio SVM segment classifiers. See `analyze_audio/readme.md` 
for instructions on how to train these audio classifiers. Currently the audio analysis module 
expects 5 audio classifiers: (1) a generic audio classifier (4-classes) (2) two speech emotion classifiers
 and (3) two musical emotion classifiers. 

## Visual
To extract hand-crafted audio features run the following:
```
python3 analyze_visual.py -f ../V236_915000__0.mp4
```
The features are saved in npy files. The main functionality is implemented in function `process_video` 
that extracts features from specific file.
See `analyze_visual/Readme.md` for more details.

You can also train a supervised model of video shots (e.g. types of shots):
```
python3 train.py -v data/class1 data/class2 -a SVM
```

The following files will be saved to disk:
 * `shot_classifier_SVM.pkl` the classifier
 * `shot_classifier_SVM_scaler.pkl` the scaler
 * `shot_classifier_SVM_results.json` the cross validation results
 * `shot_classifier_conf_mat_SVC().jpg` the confusion matrix of the cross validation procedure


As soon as the supervised model is trained you can classify an unknown shot 
(or shots organized in folders): 
```
python3 wrapper.py -m SVM -i test.mp4
```

The following script detects the change of the shots in a video file and it stores the respective shots in individual files.
It can be used in combination with the `wrapper.py` script above to analyze a movie per shot. 
```
python3 shot_generator.py -f data/file.mp4
```

## About MagCIL

This repository comes from [MagCIL](https://magcil.github.io), the Multimodal Analysis Group of the
Institute of Informatics and Telecommunications at the National Centre for Scientific Research
"Demokritos" in Athens, Greece. We work on multimodal deep learning for audio, speech, music and
video, from fundamental research to deployed systems.

- Group website: <https://magcil.github.io>
- Publications: <https://magcil.github.io/pub.html>
- All our open-source code and data: <https://magcil.github.io/code.html>
- Work with us, in industry projects or research consortia: <https://magcil.github.io/collaborate.html>

Other tools from the group:

- [`pyAudioAnalysis`](https://github.com/tyiannak/pyAudioAnalysis) &mdash; audio feature extraction, classification, segmentation and applications
- [`deep_audio_features`](https://github.com/tyiannak/deep_audio_features) &mdash; training and using CNNs for audio classification (PyTorch)
- [`deepaudio-x`](https://github.com/magcil/deepaudio-x) &mdash; a PyTorch framework for audio classification
- [`deepaudio-lab`](https://github.com/magcil/deepaudio-lab) &mdash; a no-code platform for training, evaluating and deploying audio classifiers
- [`amvoc`](https://github.com/tyiannak/amvoc) &mdash; analysis of mouse vocal communication
- [`paura`](https://github.com/tyiannak/paura) &mdash; real-time audio recording and analysis

The full list is at <https://github.com/magcil>.
