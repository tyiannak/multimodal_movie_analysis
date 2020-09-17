#!/usr/bin/python python
# -*- coding: utf-8 -*

import os
import re
import yaml


settings_filename = os.path.join(os.path.dirname(__file__), 'settings.yaml')

with open(settings_filename, "r") as f:
    settings = yaml.load(f, Loader=yaml.FullLoader)



def load_data():
    """
    Dataset loader. Will read the .srt files from a specific folders, clean the subtitles and
    return the names of the movies and the corresponding clean subtitles.
    Input:
        - nothing, will utilize the settings file for input
    Output:
        - name_of_movies: list,
        list of strings, containing the name of the movie
        - text_of_movies_cleaned: list,
        list of strings, containing the cleaned subtitles of each movie

    """
    print('~' * 50)
    name_of_movies = []
    text_of_movies_cleaned = []
    for file_ in os.listdir(settings['paths']['path_to_folder_with_srt_files']):
        if file_.endswith(".srt"):
            name_of_movies.append(os.path.basename(file_))
            cur_sub = read_srt_file(os.path.join(settings['paths']['path_to_folder_with_srt_files'], file_))
            text_of_movies_cleaned.append(clean_sub(cur_sub))
    print('#### Data Loading ####')
    print(f'Found {len(name_of_movies)} movies!')
    print('~'*50)
    print(f'#### List of movies: ####')
    for name in name_of_movies:
        print(f'{name}')
    print('~'*50)
    return name_of_movies, text_of_movies_cleaned


def read_srt_file(path_to_srt_file):
    """
    Simple functionality to read the srt file.
    Input:
        - path_to_srt_file: str,
        string with the path to the srt files
    Output:
        - text: str,
        the content of the file
    """
    with open(path_to_srt_file, 'r') as f:
        lines = f.read().splitlines()
    return " ".join(lines)


def load_stopwords(path_to_stopwords_file):
    """
    Simple loading functionality for the stopwords. These stopwords are the long english, the nltk
    and some handmade-ones.
    Input:
        - path_to_stopwords_file: str
        string with the path to the stopwords file. One stopword per line
    Output:
        - stopwords: list,
        list of the stopwords as found in the file
    """
    with open(path_to_stopwords_file, 'r') as f:
        stopwords = [line.replace("\n", "") for line in f.readlines()]
    return stopwords


def clean_sub(text):
    """
    Simple functionality to remove artifacts from the .srt file and keep the clean text.
    Input:
        - text: str,
        the string representation of the subtitles
    Output:
        - cleaned_text: str
        the cleaned string
    """
    # Create reg expressions removals
    nonan = re.compile(r'[^a-zA-Z ]')  # basically numbers
    ita_bo_re = re.compile(r'<i>|<b>|</i>|</b>')  # italics and bold
    po_re = re.compile(r'\.|\!|\?|\,|\:|\(|\)')  # punct point and others
    text = nonan.sub('', po_re.sub('', ita_bo_re.sub('', text)))
    stopwords = load_stopwords(settings["paths"]["path_to_stopwords"])
    cleaned_text = " ".join([word for word in text.lower().split() if word not in stopwords])
    return cleaned_text
