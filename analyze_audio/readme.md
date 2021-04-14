
# Audio - based movie analysis

## Train segment-level audio classifiers

### Train generic audio classifier (4-class):
This is a general audio classifier to distinguish between 4 generic classes namely speech, music, silence and others
(Note: download the 4class_balanced.zip dataset from google drive or use your own dataset of music, speech, others and silence classes):
```
python3 train.py -i 4class_balanced/music 4class_balanced/other 4class_balanced/silence 4class_balanced/speech -o audio_4class
```
Results in:
```		music			other			silence			speech		OVERALL
	C	PRE	REC	f1	PRE	REC	f1	PRE	REC	f1	PRE	REC	f1	ACC	f1
	0.001	21.9	3.1	5.4	25.0	0.0	0.0	97.2	79.5	87.4	33.9	100.0	50.6	46.5	35.9
	0.010	92.3	94.4	93.3	78.9	54.5	64.5	94.9	87.0	90.8	70.7	95.1	81.1	83.2	82.4
	0.500	94.1	97.7	95.8	88.1	83.7	85.9	94.4	94.4	94.4	92.3	93.0	92.6	92.3	92.2
	1.000	96.2	97.1	96.6	84.4	82.5	83.5	93.9	92.1	93.0	90.2	93.0	91.6	91.3	91.2
	5.000	95.7	97.7	96.7	88.1	83.7	85.9	94.5	96.1	95.3	90.9	91.6	91.3	92.4	92.3	 best f1	 best Acc
	10.000	95.4	96.5	96.0	83.1	80.9	82.0	92.4	94.8	93.6	91.5	90.5	91.0	90.8	90.6
	20.000	95.6	96.3	96.0	82.4	81.9	82.1	94.7	93.8	94.3	90.0	90.7	90.3	90.8	90.7
Confusion Matrix:
	mus	oth	sil	spe
mus	24.73	0.15	0.15	0.29
oth	0.64	19.89	1.27	1.96
sil	0.10	0.78	24.24	0.10
spe	0.39	1.76	0.00	23.56
Selected params: 5.00000
```


### Train speech emotion classifiers (2 classifiers):
Speech emotion arousal (use the `speech_analytics/data/audio/merged/overall` datasets or your own speech emotion datasets): 

```
python3 train.py -i speech/arousal/low speech/arousal/neutral speech/arousal/high -o speech_arousal
```

Results in:
```
		low			neutral			high		OVERALL
	C	PRE	REC	f1	PRE	REC	f1	PRE	REC	f1	ACC	f1
	0.001	33.3	0.0	0.0	55.7	100.0	71.6	33.3	0.0	0.0	55.7	23.9
	0.010	33.3	0.0	0.0	63.4	96.9	76.7	80.9	39.0	52.6	66.0	43.1
	0.500	74.2	17.4	28.1	71.7	93.9	81.3	79.2	61.3	69.1	73.5	59.5
	1.000	74.2	23.4	35.6	73.0	92.3	81.5	78.0	64.0	70.3	74.3	62.5
	5.000	68.3	36.8	47.9	74.3	88.8	80.9	77.5	65.7	71.1	74.7	66.6	 best f1	 best Acc
	10.000	59.6	40.0	47.9	74.5	86.0	79.9	76.3	66.0	70.8	73.7	66.2
	20.000	50.9	37.4	43.1	74.2	82.7	78.2	75.1	68.2	71.4	72.1	64.3
Confusion Matrix:
	low	neu	high
low	4.96	7.33	1.17
neu	1.52	49.50	4.71
high	0.78	9.77	20.25
Selected params: 5.00000

```

Speech emotion valence:
```
python3 train.py -i speech/valence/negative speech/valence/neutral speech/valence/positive -o speech_valence

```

results in
```
		negative			neutral			positive		OVERALL
	C	PRE	REC	f1	PRE	REC	f1	PRE	REC	f1	ACC	f1
	0.001	33.3	0.0	0.0	41.1	100.0	58.2	33.3	0.0	0.0	41.1	19.4
	0.010	49.6	58.0	53.5	52.6	67.5	59.1	33.3	0.0	0.0	51.2	37.5
	0.500	59.0	57.2	58.1	54.9	78.0	64.5	70.4	9.6	16.9	56.9	46.5
	1.000	61.4	57.6	59.4	55.9	77.2	64.9	65.8	18.8	29.2	58.5	51.2	 best Acc
	5.000	61.1	61.1	61.1	57.2	68.5	62.3	53.4	29.7	38.2	58.4	53.9
	10.000	61.3	61.4	61.4	57.5	66.1	61.5	49.9	33.1	39.8	58.1	54.2	 best f1
	20.000	58.1	58.4	58.3	54.6	59.7	57.1	45.2	35.2	39.6	54.7	51.6
Confusion Matrix:
	neg	neu	pos
neg	23.30	16.25	0.89
neu	8.43	31.73	0.92
pos	6.23	8.78	3.47
Selected params: 1.00000

```

### Train music classifiers
Music emotional energy (use the `Music/giantas_music_balanced` dataset or your own music emotion recognition dataset):
```
python3 train.py -i ~/Downloads/music/energy/low ~/Downloads/music/energy/medium ~/Downloads/music/energy/high -o music_energy
```

results in 
```
		low			medium			high		OVERALL
	C	PRE	REC	f1	PRE	REC	f1	PRE	REC	f1	ACC	f1
	0.001	33.3	0.0	0.0	33.9	100.0	50.7	33.3	0.0	0.0	33.9	16.9
	0.010	69.5	67.7	68.6	41.9	72.9	53.2	74.9	19.0	30.3	53.6	50.7
	0.500	77.6	74.7	76.1	57.7	56.7	57.2	67.5	71.3	69.4	67.5	67.5
	1.000	79.7	75.8	77.7	55.9	57.1	56.5	68.7	70.6	69.6	67.8	68.0	 best f1	 best Acc
	5.000	76.2	76.2	76.2	54.3	52.9	53.6	66.4	68.1	67.2	65.7	65.7
	10.000	73.5	73.1	73.3	53.2	53.5	53.3	68.1	68.0	68.0	64.8	64.9
	20.000	72.7	73.1	72.9	52.3	50.6	51.4	65.1	66.9	66.0	63.4	63.4
Confusion Matrix:
	low	med	high
low	25.37	6.40	1.68
med	5.74	19.39	8.82
high	0.71	8.89	23.00
Selected params: 1.00000
```

Music valence 
```
python3 train.py -i music_balanced/valence/low  music_balanced/valence/medium music_balanced/valence/high -o music_valence
```

results in 
```
		low			medium			high		OVERALL
	C	PRE	REC	f1	PRE	REC	f1	PRE	REC	f1	ACC	f1
	0.001	33.3	0.0	0.0	33.3	0.0	0.0	33.7	100.0	50.5	33.7	16.8
	0.010	57.7	68.7	62.7	25.0	0.5	1.0	47.5	84.4	60.8	51.4	41.5
	0.500	70.9	62.6	66.5	48.8	48.5	48.7	61.9	69.4	65.5	60.2	60.2	 best f1	 best Acc
	1.000	69.0	62.9	65.8	46.7	44.9	45.8	60.3	67.9	63.9	58.6	58.5
	5.000	64.6	60.6	62.5	44.8	43.6	44.2	59.2	64.3	61.6	56.2	56.1
	10.000	59.2	62.0	60.6	42.1	38.0	40.0	58.9	61.8	60.3	54.0	53.6
	20.000	59.4	61.9	60.7	42.4	40.3	41.4	59.5	60.0	59.8	54.1	53.9
Confusion Matrix:
	low	med	high
low	20.69	8.48	3.90
med	6.58	16.10	10.51
high	1.92	8.40	23.43
Selected params: 0.50000
```