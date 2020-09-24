#!/usr/bin/python !/usr/bin/env python
# -*- coding: utf-8 -*

import time
from load import settings, load_data
from process import extract_features


if __name__ == "__main__":
    time_start = time.time()
    print(f'#### Starting textual component analysis ####')
    movies, subtitles = load_data()
    print('~' * 50)
    extract_features(movies, subtitles)
    print(f'#### Finished in {time.time() - time_start:.2f} seconds ####')
