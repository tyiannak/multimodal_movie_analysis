#!/usr/bin/python
# -*- coding: utf-8 -*

import time
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from load import settings
from utils import find_similar_movies, find_top_n_similar_pairs



def extract_features(movies, subtitles):
    """
    Main function to extract features using the needed models.
    Input:
        - movies: list,
        list of strings, containing the title of the movie
        - subtitles: list,
        list of strings, containing the cleaned subtitles for each movie
    """

    modelname2func = {
        'tfidf': tfidf_model,
        'lda': lda_model,
        'nmf': nmf_model
    }
    for model, value in settings['models'].items():
        if str(value) == "True":
            try:
                time_start = time.time()
                extraction_func = modelname2func[model]
                print(f'#### Generating features using {model} model ####')
                features = extraction_func(movies, subtitles)
                print(f'#### Finished generating features using {model} model! Time: {time.time() - time_start:.2f} seconds! ####')
                print(f'#### Evaluating similarity matrix ####')
                find_top_n_similar_pairs(movies, features, 10)
                print('~' * 50)
            except KeyError as  e:
                print(f'Non-implemented feature extraction function for model {model}!')
                print(e)
                raise NotImplementedError

def save_model(path_to_model_to_be_saved, model):
    """
    Simple wrapper to save models using cPickle.
    Input:
        - path_to_model_to_be_saved: str,
        string of path to save the model to
        - model: object,
        object of some kind that we need to persist to disk
    Output:
        None, saves and returns
    """
    with open(path_to_model_to_be_saved, 'wb') as f:
        pickle.dump(model, f)


def tfidf_model(movies, subtitles):
    """
    Model to extract tfidf features from movies.
    Input:
        - movies: list,
        list of strings, containing the title of the movie
        - subtitles: list,
        list of strings, containing the cleaned subtitles for each movie
    Output:
        np.array, the feature matrix with N x F size (N movies, F features)
    """
    implemented_hyperparams = [
        'min_df',
        'max_df',
        'max_features',
        'ngram_range',
        'stop_words',
        'binary'
    ]
    conf = {}
    for hyperparam in implemented_hyperparams:
        conf[hyperparam] = settings['tfidf'][hyperparam]
    model = TfidfVectorizer(**conf)
    features = model.fit_transform(subtitles)
    feature_names = model.get_feature_names()
    df_feats = pd.DataFrame(features.toarray(), columns=feature_names)
    df_feats['Movie'] = movies
    df_feats = df_feats[['Movie'] + feature_names]
    df_feats.to_csv(settings['tfidf']['path_to_features'], index=False, header=True, encoding='utf8')
    if settings['tfidf']['save_model']:
        save_model(settings['tfidf']['path_to_model'], model)
    return df_feats[df_feats.columns[1:]].values

def nmf_model(movies, subtitles):
    """
    Model to extract nmf features from movies.
    Input:
        - movies: list,
        list of strings, containing the title of the movie
        - subtitles: list,
        list of strings, containing the cleaned subtitles for each movie
    Output:
         np.array, the feature matrix with N x F size (N movies, F features)
    """
    implemented_hyperparams = [
        'n_components',
    ]
    conf = {}
    for hyperparam in implemented_hyperparams:
        conf[hyperparam] = settings['nmf'][hyperparam]
    conf["random_state"] = settings['random_state']
    model = NMF(**conf)
    vect = TfidfVectorizer(max_df=0.7, min_df=0.2, ngram_range=(1,1), stop_words="english")
    tf_values = vect.fit_transform(subtitles)
    features = model.fit_transform(tf_values)
    feature_names = ['Topic_%d' % (i + 1) for i in range(model.n_components)]
    #print(f'Topics learned by NMF:')
    #print_top_words(model, vect.get_feature_names(), 20)
    df_feats = pd.DataFrame(features, columns=feature_names)
    df_feats['Movie'] = movies
    df_feats = df_feats[['Movie'] + feature_names]
    df_feats.to_csv(settings['nmf']['path_to_features'], index=False, header=True, encoding='utf8')
    if settings['nmf']['save_model']:
        save_model(settings['nmf']['path_to_model'], model)
    return df_feats[df_feats.columns[1:]].values



def lda_model(movies, subtitles):
    """
    Model to extract LDA features from movies.
    Input:
        - movies: list,
        list of strings, containing the title of the movie
        - subtitles: list,
        list of strings, containing the cleaned subtitles for each movie
    Output:
         np.array, the feature matrix with N x F size (N movies, F features)
    """
    implemented_hyperparams = [
        'n_components',
    ]
    conf = {}
    for hyperparam in implemented_hyperparams:
        conf[hyperparam] = settings['lda'][hyperparam]
    conf["random_state"] = settings['random_state']
    model = LatentDirichletAllocation(**conf)
    vect = CountVectorizer(max_df=0.7, min_df=0.2, ngram_range=(1,1), stop_words="english")
    count_values = vect.fit_transform(subtitles)
    features = model.fit_transform(count_values)
    feature_names = ['Topic_%d' % (i + 1) for i in range(model.n_components)]
    #print(f'Topics learned by LDA:')
    #print_top_words(model, vect.get_feature_names(), 20)
    df_feats = pd.DataFrame(features, columns=feature_names)
    df_feats['Movie'] = movies
    df_feats = df_feats[['Movie'] + feature_names]
    df_feats.to_csv(settings['lda']['path_to_features'], index=False, header=True, encoding='utf8')
    if settings['lda']['save_model']:
        save_model(settings['lda']['path_to_model'], model)
    return df_feats[df_feats.columns[1:]].values


def print_top_words(model, feature_names, n_top_words=20):
    """
    Helper functionality for printing the topics.
    Taken from:
    https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
    Input:
        - model: obj,
        usually NMF or LDA object from sklearn
        - features: list,
        list of the names of the tokens of the BoW feature-space
        - n_top_words: int,
        number of words per topic to show
    """
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
